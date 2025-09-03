"""
无不可达区域、弱风（常量风矢量）
固定需求集（在t=0全部已知）
单架无人机（用于往返能耗与时耗校准）
事件驱动：仅在无人机空闲时进行一次指派；飞行到达、服务完成、返仓等均通过事件推进
能耗模型与参数采用默认数值
"""

# %%
import math
import random
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


# %% 配置与常量
CONFIG = {
    # 区域与单位
    "area_km": 20.0,                # 区域正方形边长（km）
    "depot_xy_km": (10.0, 10.0),    # 仓库坐标（km）
    "wind_mps": (2.0, 0.5),         # 风场（弱风常量）(wx, wy) m/s
    # 无人机参数
    "uav_speed_mps": 15.0,          # 巡航速度（m/s）
    "uav_payload_kg": 5.0,          # 最大载重（kg）
    "battery_wh": 600.0,            # 电池容量（Wh）
    "battery_safe_ratio": 0.8,      # 安全可用容量比例（去-服-返）
    "recharge_time_s": 180,         # 仓库换电/补给时间（s）
    # 能耗参数
    # 飞行能耗（Wh/km）= 12 + 3*载重（kg）+ 0.8*逆风风量(m/s)，下线裁剪为8
    "base_flight_wh_per_km": 12.0,
    "perkg_flight_wh_per_km": 3.0,
    "headwind_coef_wh_per_km_per_mps": 0.8,
    "min_flight_wh_per_km": 8.0,
    # 服务能耗（Wh/min）= 15 + 3*载重（kg）
    "base_service_wh_per_min": 15.0,
    "perkg_service_wh_per_min": 3.0,
    # 服务时间（min）= 1.0 + 0.5*投放量（kg）
    "service_time_offset_min": 1.0,
    "service_time_perkg_min": 0.5,
    # 需求生成（固定集）
    "num_demands": 20,              # 固定需求数量
    "demand_kg_minmax": (1.0, 3.0), # 投放量范围（kg）
    "seed": 200405081012,           # 随机种子
    # 调度策略
    "policy": "greedy_nearest",     # 简化策略策略：贪心最近
    # 日志
    "verbose": True,
}

# %% 工具函数
def km_to_m(km: float) -> float:
    """公里转米"""
    return km * 1000.0

def m_to_km(m: float) -> float:
    """米转公里"""
    return m / 1000.0

def minutes_to_seconds(mins: float) -> float:
    """分钟转秒"""
    return mins * 60.0

def distance_m(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """计算二维平面上两点间的欧氏距离（米）
    Args:
        p: 点1坐标 (x, y) 米
        q: 点2坐标 (x, y) 米
    Returns:
        欧氏距离（米）
    """
    dx, dy = p[0] - q[0], p[1] - q[1]
    return math.hypot(dx, dy)

def unit_vector(p: Tuple[float, float], q: Tuple[float, float]) -> Tuple[float, float]:
    """计算二维平面上两点间的单位方向向量
    Args:
        p: 起点坐标 (x, y)
        q: 终点坐标 (x, y)
    Returns:
        单位方向向量 (ux, uy)
    """
    d = distance_m(p, q)
    if d == 0:
        return (0.0, 0.0)
    return ((q[0] - p[0]) / d, (q[1] - p[1]) / d)

def headwind_component_mps(dir_unit: Tuple[float, float], wind_mps: Tuple[float, float]) -> float:
    """计算给定方向上的逆风分量（米/秒）
    Args:
        dir_unit: 方向单位向量 (ux, uy)
        wind_mps: 风矢量 (wx, wy) 米/秒
    Returns:
        逆风分量（米/秒），无逆风时为0
    """
    # 逆风 = max(0, -dot(wind, dir))
    dot = wind_mps[0] * dir_unit[0] + wind_mps[1] * dir_unit[1]
    return max(0.0, -dot)

# %% 数据类
@dataclass
class Demand:
    """需求数据类"""
    did: int                    # 需求ID
    xy_m: Tuple[float, float]   # 需求坐标（米）
    quantity_kg: float          # 投放量（kg）
    served_kg: float = 0.0      # 已服务量（kg）

    def remaining_kg(self) -> float:
        """剩余量 = max(0, 投放量 - 已服务量)"""
        return max(0.0, self.quantity_kg - self.served_kg)

    def is_done(self) -> bool:
        """是否已完成服务"""
        return self.remaining_kg() <= 1e-8

@dataclass
class UAV:
    """无人机数据类"""
    uid: int                            # 无人机ID
    xy_m: Tuple[float, float]           # 当前位置（米）
    speed_mps: float                    # 巡航速度（米/秒）
    payload_cap_kg: float               # 最大载重（kg）
    battery_wh: float                   # 电池容量（Wh）
    battery_full_wh: float              # 电池满容量（Wh）
    safe_ratio: float                   # 安全可用容量比例
    status: str = "idle"                # 状态：idle, enroute_to_task, servicing, returning, recharging carrying
    carrying_kg: float = 0.0            # 当前载重（kg）
    current_task: Optional[int] = None  # 当前任务ID（需求ID）
    total_flight_time_s: float = 0.0    # 累计飞行时间（秒）
    total_service_time_s: float = 0.0   # 累计服务时间（秒）
    total_distance_m: float = 0.0       # 累计飞行距离（米）
    total_energy_wh: float = 0.0        # 累计能耗（Wh）
    completed_tasks: int = 0            # 完成任务数

    # 可用能量（Wh）
    def available_energy_wh(self) -> float:
        """可用能量 = 当前电量"""
        return self.battery_wh

    # 安全可用能量（Wh）
    def usable_energy_wh(self) -> float:
        """安全可用能量 = 电池满容量 * 安全比例"""
        return self.battery_full_wh * self.safe_ratio

    def reset_to_depot(self, depot_xy_m: Tuple[float, float]):
        """重置无人机位置与状态到仓库
        Args:
            depot_xy_m: 仓库坐标（米）
        """
        self.xy_m = depot_xy_m
        self.status = "idle"
        self.carrying_kg = 0.0
        self.current_task = None

# %% 能耗与时间模型
class EnergyModel:
    """能耗类"""
    def __init__(self, config: Dict):
        self.base_flight_wh_per_km = config["base_flight_wh_per_km"]
        self.perkg_flight_wh_per_km = config["perkg_flight_wh_per_km"]
        self.headwind_coef_wh_per_km_per_mps = config["headwind_coef_wh_per_km_per_mps"]
        self.min_flight_wh_per_km = config["min_flight_wh_per_km"]
        self.base_service_wh_per_min = config["base_service_wh_per_min"]
        self.perkg_service_wh_per_min = config["perkg_service_wh_per_min"]
        self.service_time_offset_min = config["service_time_offset_min"]
        self.service_time_perkg_min = config["service_time_perkg_min"]
    
    def flight_energy_wh(self, dist_m: float, load_kg: float, headwind_mps: float) -> float:
        """计算飞行能耗（Wh）
        Args:
            dist_m: 飞行距离（米）
            load_kg: 载重（kg）
            headwind_mps: 逆风分量（米/秒）
        Returns:
            飞行能耗（Wh）
        """
        dist_km = m_to_km(dist_m)
        wh_per_km = max(self.min_flight_wh_per_km,
                        self.base_flight_wh_per_km + self.perkg_flight_wh_per_km * load_kg + self.headwind_coef_wh_per_km_per_mps*headwind_mps)
        return wh_per_km * dist_km
    
    def flight_time_s(self, dist_m: float, speed_mps: float) -> float:
        """计算飞行时间（秒）
        Args:
            dist_m: 飞行距离（米）
            speed_mps: 飞行速度（米/秒）
        Returns:
            飞行时间（秒）
        """
        return dist_m / max(speed_mps, 1e-8)
    
    def service_time_s(self, deliver_kg: float) -> float:
        """计算服务时间（秒）
        Args:
            deliver_kg: 投放量（kg）
        Returns:
            服务时间（秒）
        """
        time_min = self.service_time_offset_min + self.service_time_perkg_min * deliver_kg
        return minutes_to_seconds(time_min)
    
    def service_energy_wh(self, deliver_kg: float) -> float:
        """计算服务能耗（Wh）
        Args:
            deliver_kg: 投放量（kg）
        Returns:
            服务能耗（Wh）
        """
        time_min = self.service_time_offset_min + self.service_time_perkg_min * deliver_kg
        wh_per_min = self.base_service_wh_per_min + self.perkg_service_wh_per_min * deliver_kg
        return wh_per_min * time_min

# %% 事件相关
@dataclass(order=True)
class Event:
    """事件数据类"""
    time_s: float
    seq: int
    etype: str = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)

# %% 环境类
class SimplifiedEnv:
    """简化事件驱动环境类"""
    def __init__(self, config: Dict):
        self.config = config
        self.random = random.Random(config["seed"])
        self.area_m = km_to_m(config["area_km"])
        depot_km = config["depot_xy_km"]
        self.depot_xy_m = tuple(km_to_m(c) for c in depot_km)
        self.wind_mps = config["wind_mps"]

        # 生成固定需求集
        self.demands: List[Demand] = self._generate_demands()
        # 创建单机
        self.uav = UAV(
            uid=0,
            xy_m=self.depot_xy_m,
            speed_mps=config["uav_speed_mps"],
            payload_cap_kg=config["uav_payload_kg"],
            battery_wh=config["battery_wh"],
            battery_full_wh=config["battery_wh"],
            safe_ratio=config["battery_safe_ratio"],
        )
        self.energy_model = EnergyModel(config)

        # 事件队列
        self.now_s = 0.0
        self._event_seq = 0
        self.event_q: List[Event] = []

        # 统计
        self.log: List[str] = []

    def _log(self, msg: str):
        """日志记录"""
        if self.config["verbose"]:
            print(f"[{self.now_s:8.1f}s] {msg}")
        self.log.append(f"[{self.now_s:8.1f}s] {msg}")
    
    def _generate_demands(self) -> List[Demand]:

        n = self.config["num_demands"]
        kg_min, kg_max = self.config["demand_kg_minmax"]
        demands = []
        for i in range(n):
            # 均匀散布在区域内
            x = self.random.uniform(0, self.area_m)
            y = self.random.uniform(0, self.area_m)
            # 避免离仓库极近/极远，略作约束
            # 保证至少0.5km远，至多9.0km远
            d = distance_m(self.depot_xy_m, (x, y))
            if d < km_to_m(0.5) or d > km_to_m(9.0):
                x = min(max(self.depot_xy_m[0] + self.random.uniform(-km_to_m(9.0), km_to_m(9.0)), 0), self.area_m)
                y = min(max(self.depot_xy_m[1] + self.random.uniform(-km_to_m(9.0), km_to_m(9.0)), 0), self.area_m)
            quantity_kg = self.random.uniform(kg_min, kg_max)
            demands.append(Demand(did=i, xy_m=(x, y), quantity_kg=quantity_kg))
        return demands
    
    def _push_event(self, time_s: float, etype: str, payload: Dict):
        """推入事件队列
        Args:
            time_s: 事件时间（秒）
            etype: 事件类型
            payload: 事件负载
        """
        self._event_seq += 1
        heapq.heappush(self.event_q, Event(time_s, self._event_seq, etype, payload))

    def _pop_event(self) -> Optional[Event]:
        """弹出事件队列顶端事件"""
        if not self.event_q:
            return None
        event = heapq.heappop(self.event_q)
        self.now_s = event.time_s
        return event
    
    def _fly(self, start_xy: Tuple[float, float], end_xy: Tuple[float, float], load_kg: float) -> Tuple[float, float]:
        """计算飞行时间与能耗
        Args:
            start_xy: 起点坐标（米）
            end_xy: 终点坐标（米）
            load_kg: 载重（kg）
        Returns:
            (飞行时间（秒）, 飞行能耗（Wh）)
        """
        dist_m = distance_m(start_xy, end_xy)
        dir_unit = unit_vector(start_xy, end_xy)
        headwind_mps = headwind_component_mps(dir_unit, self.wind_mps)
        t_s = self.energy_model.flight_time_s(dist_m, self.uav.speed_mps)
        e_wh = self.energy_model.flight_energy_wh(dist_m, load_kg, headwind_mps)
        return t_s, e_wh
    
    def _service(self, deliver_kg: float) -> Tuple[float, float]:
        """计算服务时间与能耗
        Args:
            deliver_kg: 投放量（kg）
        Returns:
            (服务时间（秒）, 服务能耗（Wh）)
        """
        t_s = self.energy_model.service_time_s(deliver_kg)
        e_wh = self.energy_model.service_energy_wh(deliver_kg)
        return t_s, e_wh

    def _estimate_roundtrip_energy(self, demand: Demand, deliver_kg: float) -> float:
        """估计往返能耗（Wh）
        Args:
            demand: 需求对象
            deliver_kg: 投放量（kg）
        Returns:
            往返能耗（Wh）
        """
        e_total = 0.0
        # 去程
        t_go, e_go = self._fly(self.depot_xy_m, demand.xy_m, deliver_kg)
        # 服务
        t_service, e_service = self._service(deliver_kg)
        # 返程
        t_return, e_return = self._fly(demand.xy_m, self.depot_xy_m, 0.0)
        e_total = e_go + e_service + e_return
        return e_total
    
    def _feasible(self, demand: Demand, deliver_kg: float) -> bool:
        """检查投放量的可行性
        Args:
            demand: 需求对象
            deliver_kg: 投放量（kg）
        Returns:
            可行性（True/False）"""
        if deliver_kg <= 0:
            return False
        if deliver_kg > self.uav.payload_cap_kg:
            return False
        if demand.is_done():
            return False
        # 估计往返能耗
        est_e = self._estimate_roundtrip_energy(demand, deliver_kg)
        return est_e <= self.uav.usable_energy_wh() and est_e <= self.uav.available_energy_wh()
    
    def _select_next_task(self) -> Optional[Tuple[int, float]]:
        """选择下一个任务（贪心最近）
        Returns:
            (需求ID, 投放量（kg）) 或 None
        """
        candidates = []
        for d in self.demands:
            if d.is_done():
                continue
            deliver = min(d.remaining_kg(), self.uav.payload_cap_kg)
            if self._feasible(d, deliver):
                dist = distance_m(self.uav.xy_m, d.xy_m)
                candidates.append((dist, d.did, deliver))
        if not candidates:
            return None
        # 贪心最近
        candidates.sort(key=lambda x: x[0])
        _, did, deliver = candidates[0]
        return did, deliver
    
    def _handle_decision(self):
        """处理无人机空闲时的决策"""
        assert self.uav.status == "idle"
        choice = self._select_next_task()
        if choice is None:
            # 若任务均不可行但有未完成需求，尝试充电；否则结束
            if any(not d.is_done() for d in self.demands):
                if self.uav.battery_wh < self.uav.usable_energy_wh():
                    self._log("No feasible task now; start recharge.")
                    self.uav.status = "recharging"
                    self._push_event(self.now_s + self.config["recharge_time_s"], "recharge_done", {})
                else:
                    self._log("No feasible task even with sufficient battery; all remaing demands exceed load or unreachable.")
                    # 无法完成剩余任务，结束
            else:
                self._log("All demands served; Simulation can stop.")
            return None
        
        did, deliver = choice
        demand = self.demands[did]
        self.uav.status = "enroute_to_task"
        self.uav.carrying_kg = deliver
        self.uav.current_task = did

        # 计算飞行时间与能耗
        t_go, e_go = self._fly(self.uav.xy_m, demand.xy_m, deliver)
        # 记录并发出到达事件
        self._log(f"Dispatch to demand#{did}: deliver {deliver:.2f}kg | fly_go t={t_go:.1f}s e={e_go:.1f}Wh")
        # 资源更新延后至事件实际发生；这里仅安排事件
        self._push_event(self.now_s + t_go, "arrive_task", {"did": did, "e_go": e_go, "t_go": t_go})
    
    def _handle_arrive_task(self, payload: dict):
        """处理到达任务事件
        Args:
            payload: 事件负载
        """
        did = payload["did"]
        e_go = payload["e_go"]
        t_go = payload["t_go"]
        demand = self.demands[did]
        # 去程更新
        self.uav.xy_m = demand.xy_m
        self.uav.battery_wh -= e_go
        self.uav.total_energy_wh += e_go
        self.uav.total_flight_time_s += t_go
        self.uav.total_distance_m += distance_m(self.depot_xy_m, demand.xy_m)
        # 服务
        deliver = min(self.uav.carrying_kg, demand.remaining_kg())
        t_sv, e_sv = self._service(deliver)
        self._log(f"Arrived demand#{did}: service deliver {deliver:.2f}kg | t={t_sv:.1f}s e={e_sv:.1f}Wh")
        # 服务更新
        demand.served_kg += deliver
        self.uav.battery_wh -= e_sv
        self.uav.total_energy_wh += e_sv
        self.uav.total_service_time_s += t_sv
        self.uav.status = "servicing"
        # 完成服务后安排返仓事件
        self._push_event(self.now_s + t_sv, "service_done", {"did": did, "delivered": deliver})

    def _handle_service_done(self, payload: dict):
        """处理服务完成事件
        Args:
            payload: 事件负载
        """
        did = payload["did"]
        delivered = payload["delivered"]
        self.uav.status = "returning"
        # 返程
        t_rt, e_rt = self._fly(self.uav.xy_m, self.depot_xy_m, 0.0)
        self._log(f"Service done demand#{did}, returning | fly_back t={t_rt:.1f}s e={e_rt:.1f}Wh")
        self._push_event(self.now_s + t_rt, "arrive_depot", {"did": did, "delivered": delivered, "e_rt": e_rt, "t_rt": t_rt})

    def _handle_arrive_depot(self, payload: dict):
        """处理到达仓库事件
        Args:
            payload: 事件负载
        """
        did = payload["did"]
        delivered = payload["delivered"]
        e_rt = payload["e_rt"]
        t_rt = payload["t_rt"]
        # 返程更新
        self.uav.xy_m = self.depot_xy_m
        self.uav.battery_wh -= e_rt
        self.uav.total_energy_wh += e_rt
        self.uav.total_flight_time_s += t_rt
        self.uav.total_distance_m += distance_m(self.demands[did].xy_m, self.depot_xy_m)
        self.uav.completed_tasks += 1
        self.uav.status = "idle"
        self.uav.carrying_kg = 0.0
        self.uav.current_task = None

        self._log(f"Arrived depot from demand#{did}, delivered {delivered:.2f}kg. Battery now {self.uav.battery_wh:.1f}Wh")
        # 到仓库后，检查是否需要充电；否则继续决策
        if self.uav.battery_wh < 0.5 * self.uav.battery_full_wh:
            self._log("Battery low, start recharge.")
            self.uav.status = "recharging"
            self._push_event(self.now_s + self.config["recharge_time_s"], "recharge_done", {})
        else:
            # 立即决策下一个任务
            self._push_event(self.now_s, "decision", {})
    
    def _handle_recharge_done(self, payload: dict):
        """处理充电完成事件
        Args:
            payload: 事件负载
        """
        self.uav.battery_wh = self.uav.battery_full_wh
        self.uav.status = "idle"
        self._log(f"Recharge done. Battery full at {self.uav.battery_wh:.1f}Wh")
        # 立即决策下一个任务
        self._push_event(self.now_s, "decision", {})
    
    def reset(self):
        """重置环境"""
        self.now_s = 0.0
        self._event_seq = 0
        self.event_q.clear()
        self.log.clear()
        self.uav.reset_to_depot(self.depot_xy_m)
        self.uav.battery_wh = self.uav.battery_full_wh
        self.uav.total_flight_time_s = 0.0
        self.uav.total_service_time_s = 0.0
        self.uav.total_distance_m = 0.0
        self.uav.total_energy_wh = 0.0
        self.uav.completed_tasks = 0
        # 重新生成需求集
        self.demands = self._generate_demands()
        # 推入初始决策事件
        self._push_event(0.0, "decision", {})
    
    def step_until_done(self, max_time_s: float = 6*3600.0):
        """运行环境直到完成或超时
        Args:
            max_time_s: 最大运行时间（秒）
        """
        while self.event_q:
            event = self._pop_event()
            if event is None:
                break
            if self.now_s > max_time_s:
                self._log("Time limit reached. Stop.")
                break
            if event.etype == "decision":
                self._handle_decision()
            elif event.etype == "arrive_task":
                self._handle_arrive_task(event.payload)
            elif event.etype == "service_done":
                self._handle_service_done(event.payload)
            elif event.etype == "arrive_depot":
                self._handle_arrive_depot(event.payload)
            elif event.etype == "recharge_done":
                self._handle_recharge_done(event.payload)
            else:
                raise ValueError(f"Unknown event type: {event.etype}")

            # 结束条件：所有需求完成且无人机空闲或充电
            if all(d.is_done() for d in self.demands) and self.uav.status in ("idle", "recharging"):
                # 若idle且无后续决策事件，则结束
                if not any(e.etype == "decision" for e in self.event_q):
                    self._log("All demands served and UAV idle/recharging; Simulation ends.")
                    break
    
    def summary(self) -> Dict:
        """生成统计摘要
        Returns:
            统计摘要字典
        """
        total_demand = sum(d.quantity_kg for d in self.demands)
        served = sum(d.served_kg for d in self.demands)
        return {
            "simulation_time_s": self.now_s,
            "total_demand_kg": total_demand,
            "served_kg": served,
            "served_ratio": served / total_demand if total_demand > 0 else 1.0,
            "uav_flight_time_s": self.uav.total_flight_time_s,
            "uav_service_time_s": self.uav.total_service_time_s,
            "uav_distance_km": m_to_km(self.uav.total_distance_m),
            "uav_energy_wh": self.uav.total_energy_wh,
            "uav_completed_tasks": self.uav.completed_tasks,    
        }

    def calibrate_one_trip(self, out_km: float, deliver_kg: float) -> Dict:
        """校准单次往返飞行与服务的时间与能耗
        Args:
            out_km: 去程距离（公里）
            deliver_kg: 投放量（kg）
        Returns:
            校准结果字典
        """
        start = self.depot_xy_m
        target = (self.depot_xy_m[0] + km_to_m(out_km), self.depot_xy_m[1])
        # 去程
        t_go, e_go = self._fly(start, target, deliver_kg)
        # 服务
        t_sv, e_sv = self._service(deliver_kg)
        # 返程
        t_rt, e_rt = self._fly(target, start, 0.0)
        return {
            "out_km": out_km,
            "deliver_kg": deliver_kg,
            "t_go_s": t_go, "e_to_wh": e_go,
            "t_sv_s": t_sv, "e_sv_wh": e_sv,
            "t_rt_s": t_rt, "e_rt_wh": e_rt,
            "t_total_s": t_go + t_sv + t_rt,
            "t_total_wh": e_go + e_sv + e_rt,
        }

def main():
    env = SimplifiedEnv(CONFIG)
    # 校准：出航距离5km，投放2kg
    calib = env.calibrate_one_trip(out_km=5.0, deliver_kg=2.0)
    print("Calibration for one trip (5km out, 2kg deliver):")
    for k, v in calib.items():
        if isinstance(v, float):
            print(f"\t{k}: {v:.2f}")
        else:
            print(f"\t{k}: {v}")
    print("\nStarting simulation...")
    
    # 重置并运行
    env.reset()
    env.step_until_done(max_time_s=6*3600.0)
    print("\nSimulation done. Summary:")
    summary = env.summary()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"\t{k}: {v:.2f}")
        else:
            print(f"\t{k}: {v}")