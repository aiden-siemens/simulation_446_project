import numpy as np
import math
from collections import deque, defaultdict
import copy

"""
Project: DES: Tim Hortons Operations
File: 446_proj.py

OVERVIEW:
This script implements a discrete-event simulation (DES) to model the daily operations of a 
Quick Service Restaurant (QSR). It simulates customer flow across three channels (Walk-in, 
Drive-thru, Mobile) through ordering, kitchen production, and fulfillment phases. The goal 
is to evaluate operational performance and net profit under stochastic demand.

KEY FEATURES:
1.  Multi-Channel Logic: Handles distinct behaviors for Walk-in (M/M/c), Drive-thru (M/M/1 
    with Balking), and Mobile (Virtual Queue with Reneging).
2.  Integrated Kitchen: Models parallel production lines (Urn, Espresso, Food) merging into 
    a synchronized Packing station with blocking logic.
3.  Stochastic Events: Simulates random machine breakdowns (Espresso Maintenance) and 
    time-varying arrival rates (Peak/Lull periods).
4.  Statistical Rigor: Uses Common Random Numbers (CRN) across 5 independent replications 
    to calculate 95% Confidence Intervals for profit and wait times.

CONFIGURATION:
-   Staffing levels, Queue Capacities, and Menu Costs are defined in the 'CONFIGURATION' 
    section at the top of the script.
-   Simulation runs for a 16-hour operating day (1020 minutes).

USAGE:
Run the script directly via Python interpreter.

OUTPUT:
The script prints a detailed event log for each replication, followed by a final 
'CROSS-REPLICATION SUMMARY' table containing Mean values and Confidence Intervals 
for Net Profit, Throughput, and Tail Latency.
"""
# 1. CONFIGURATION
SIM_DURATION = 16 * 60
CLOSE_TIME = 16 * 60
MORNING_TIME = 4.0 * 60
LUNCH_TIME = 8.0 * 60

# Staffing
NUM_CASHIERS = 1
NUM_COOKS = 2
NUM_BARISTAS = 2
NUM_PACKERS = 1
NUM_BUSSERS = 1

# Capacities
SHELF_CAPACITY = 3
DRIVE_THRU_LANE_SIZE = 7
URN_BATCH_SIZE = 50
NUM_TABLES = 10

MENU_CATALOG = {
    'coffee':     {'price': 2.30, 'cost': 0.15, 'time': 0.2},
    'tea':        {'price': 2.30, 'cost': 0.15, 'time': 0.2},
    'espresso':   {'price': 3.50, 'cost': 0.15, 'time': 0.5},
    'latte':      {'price': 4.50, 'cost': 0.70, 'time': 0.5},
    'cappuccino': {'price': 4.50, 'cost': 0.70, 'time': 0.5},
    'bagel':      {'price': 3.00, 'cost': 0.40, 'time': 1.0},
    'breakfast':  {'price': 4.50, 'cost': 0.80, 'time': 1.0},
    'wrap':       {'price': 7.50, 'cost': 1.00, 'time': 1.0},
    'sandwich':   {'price': 7.50, 'cost': 1.00, 'time': 1.0},
    'donut':      {'price': 1.50, 'cost': 0.20, 'time': 0.2},
}

MENU_KEYS = list(MENU_CATALOG.keys())

# Pack Time Constants
ORDER_TIME = 0.5
PACK_PER_ITEM = 0.15
CLEAN_TIME = 0.5

HOURLY_WAGE = 17.65

# Global Constants for arrival_type
WALK = 1; DRIVE = 2; MOBILE = 3
PROB_DINE_IN = 0.5
LOSS_LOG = [] 

# 2. CUSTOMER CLASS
class Customer:
    def __init__(self, c_id, arrival_type, order, arrival_time):
        self.id = c_id
        self.arrival_type = arrival_type
        self.type_str = "WALK" if arrival_type == 1 else ("DRIVE" if arrival_type == 2 else "MOBILE")
        self.order = order
        self.arrival_time = arrival_time
        
        self.is_dine_in = False
        if arrival_type == WALK and np.random.rand() < PROB_DINE_IN:
            self.is_dine_in = True
            self.type_str = "WALK (DINE)"

        self.order_value = sum(MENU_CATALOG[item]['price'] for item in self.order)
        self.order_cost = sum(MENU_CATALOG[item]['cost'] for item in self.order)

        self.departure_time = 0 
        self.kitchen_entry_time = 0.0 
        self.kitchen_finish_time = 0.0
        
        self.is_packed = False 
        self.promised_time = 0 
        if arrival_type == MOBILE:
            self.promised_time = arrival_time + np.random.uniform(10, 20)
            
        self.queue_delays = {'ORDER': 0.0, 'URN': 0.0, 'ESP': 0.0, 'FOOD': 0.0, 'PACK': 0.0, 'SEAT': 0.0}
        self.entry_times = {}

    def __repr__(self):
        return f"[#{self.id} {self.type_str}]"

# --- 3. INPUT GENERATOR ---
class ScenarioGenerator:
    def __init__(self, seed):
        np.random.seed(seed)
    
    def generate_workload(self):
        walk = self._gen_stream(WALK, 3.0)
        drive = self._gen_stream(DRIVE, 2.0)
        mobile = self._gen_stream(MOBILE, 4.0)
        
        pooled = walk + drive + mobile
        pooled.sort(key=lambda x: x.arrival_time)
        return pooled

    def _gen_stream(self, c_type, base_rate):
        stream = []
        clock = 0.0
        count = 0
        while clock < CLOSE_TIME:
            rate = base_rate
            if clock < MORNING_TIME: rate *= 1.5
            elif clock < LUNCH_TIME: rate *= 0.6
            else: rate *= 1.2
            
            inter = -rate * np.log(np.random.uniform(0, 1))
            clock += inter
            if clock >= CLOSE_TIME: break
            
            count += 1
            num_items = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.25, 0.2, 0.15, 0.1 ])
            order = np.random.choice(MENU_KEYS, size=num_items, replace=True).tolist()
            c_id = f"{'W' if c_type==1 else ('D' if c_type==2 else 'M')}-{count}"
            stream.append(Customer(c_id, c_type, order, clock))
        return stream

# 4. INTEGRATED SIMULATION
class RestaurantSimulation:
    def __init__(self, workload):
        self.sim_time = 0.0
        self.incoming_customers = deque(workload)
        
        # Queues
        self.q_order_walk = deque()
        self.q_order_drive = deque()   
        self.dt_window_queue = deque() 
        
        self.q_urn = deque(); self.q_esp = deque(); self.q_food = deque()
        self.q_pack = deque(); self.shelf = deque()
        self.q_seating = deque()

        # Resources
        self.busy_cashiers = 0
        self.busy_speaker = 0
        
        self.busy_urn_count = 0; self.busy_esp_count = 0
        self.busy_cook_count = 0; self.busy_packers = []
        self.busy_busser_count = 0
        
        # State
        self.drive_lane_count = 0 
        self.urn_cups = URN_BATCH_SIZE; self.urn_brewing = False
        self.esp_maintenance = False # NEW: Maintenance State
        self.tables_free = NUM_TABLES; self.tables_dirty = 0
        
        self.active_kitchen_orders = {}
        self.completed_list = []
        self.shelf_max_hits = 0 
        self.total_busy_time = {'URN': 0.0, 'ESP': 0.0, 'FOOD': 0.0, 'PACK': 0.0, 'SEAT': 0.0, 'BUS': 0.0}
        
        # Events
        self.order_finish_events = [] 
        self.urn_finish_events = []
        self.esp_finish_events = []
        self.cook_finish_events = [] 
        self.pack_finish_events = []
        self.mobile_arrival_events = []
        self.eating_finish_events = []
        self.cleaning_finish_events = []
        self.maint_finish_event = []

        # EVENT LIST
        # [12] Maint Start, [13] Maint End
        self.time_next_event = [0.0] * 14
        self.num_events = 13
        for i in range(2, 14): self.time_next_event[i] = float('inf')
        
        if self.incoming_customers:
            self.time_next_event[1] = self.incoming_customers[0].arrival_time
        else:
            self.time_next_event[1] = float('inf')

        # Allocate for random maintenance
        maint_1 = np.random.uniform(0, MORNING_TIME)
        maint_2 = np.random.uniform(0, CLOSE_TIME)
        
        self.maintenance_schedule = sorted([maint_1, maint_2])
        self.time_next_event[12] = self.maintenance_schedule.pop(0)

    def run(self):
        print(f"\n--- STARTING INTEGRATED SIMULATION (LANE CAP: {DRIVE_THRU_LANE_SIZE}) ---")
        while True:
            self._update_event(2, self.order_finish_events)
            self._update_event(3, self.urn_finish_events)
            self._update_event(4, self.esp_finish_events)
            self._update_event(5, self.cook_finish_events)
            self._update_event(6, self.pack_finish_events)
            self._update_event(9, self.mobile_arrival_events)
            self._update_event(10, self.eating_finish_events)
            self._update_event(11, self.cleaning_finish_events)

            min_time = float('inf')
            next_type = 0
            for i in range(1, self.num_events + 1):
                if self.time_next_event[i] < min_time:
                    min_time = self.time_next_event[i]
                    next_type = i
            
            if next_type == 0: break
            self.sim_time = min_time
            
            # Dispatch
            if next_type == 1: self.handle_arrival()
            elif next_type == 2: self.handle_order_done()
            elif next_type == 3: self.handle_urn_done()
            elif next_type == 4: self.handle_esp_done()
            elif next_type == 5: self.handle_food_done()
            elif next_type == 6: self.handle_pack_done()
            elif next_type == 7: self.handle_pickup()
            elif next_type == 8: self.handle_urn_refill()
            elif next_type == 9: self.handle_mobile_check()
            elif next_type == 10: self.handle_eating_done()
            elif next_type == 11: self.handle_cleaning_done()
            elif next_type == 12: self.handle_maint_start()
            elif next_type == 13: self.handle_maint_end()

    def _update_event(self, idx, event_list):
        if event_list:
            event_list.sort(key=lambda x: x[0])
            self.time_next_event[idx] = event_list[0][0]
        else:
            self.time_next_event[idx] = float('inf')

    def expon(self, mean):
        return -mean * np.log(np.random.uniform(0, 1))
    
    def get_service_time(self, base):
        return base + self.expon(0.25)

    def calculate_eating_time(self, cust):
        eating_time = 10.0 
        for item in cust.order:
            if item in ['bagel', 'sandwich']: eating_time += 5.0
            elif item in ['coffee', 'tea']:   eating_time += 3.0
            else:                             eating_time += 2.0
        return self.expon(eating_time)

    # handlers
    def handle_arrival(self):
        cust = self.incoming_customers.popleft()
        
        if self.incoming_customers:
            self.time_next_event[1] = self.incoming_customers[0].arrival_time
        else:
            self.time_next_event[1] = float('inf')

        if cust.arrival_type == DRIVE:
            if self.drive_lane_count >= DRIVE_THRU_LANE_SIZE:
                LOSS_LOG.append({'time': self.sim_time, 'cust': cust, 'reason': 'BALK (Lane Full)'})
                return 
            self.drive_lane_count += 1
            self.q_order_drive.append(cust)
            cust.entry_times['ORDER'] = self.sim_time
            self.try_order_drive()

        elif cust.arrival_type == WALK:
            self.q_order_walk.append(cust)
            cust.entry_times['ORDER'] = self.sim_time
            self.try_order_walk()

        elif cust.arrival_type == MOBILE:
            self.send_to_kitchen(cust)
            self.mobile_arrival_events.append((cust.promised_time, cust))

    def try_order_drive(self):
        if self.busy_speaker == 0 and self.q_order_drive:
            cust = self.q_order_drive.popleft()
            self.busy_speaker = 1
            cust.queue_delays['ORDER'] += (self.sim_time - cust.entry_times['ORDER'])
            dur = self.get_service_time(ORDER_TIME)
            self.order_finish_events.append((self.sim_time + dur, cust))

    def try_order_walk(self):
        if self.busy_cashiers < NUM_CASHIERS and self.q_order_walk:
            cust = self.q_order_walk.popleft()
            self.busy_cashiers += 1
            cust.queue_delays['ORDER'] += (self.sim_time - cust.entry_times['ORDER'])
            dur = self.get_service_time(ORDER_TIME)
            self.order_finish_events.append((self.sim_time + dur, cust))

    def handle_order_done(self):
        _, cust = self.order_finish_events.pop(0)
        if cust.arrival_type == DRIVE:
            self.busy_speaker = 0
            self.dt_window_queue.append(cust)
            self.try_order_drive() 
        else:
            self.busy_cashiers -= 1
            self.try_order_walk()
        self.send_to_kitchen(cust)

    def send_to_kitchen(self, cust):
        cust.kitchen_entry_time = self.sim_time
        urns = [i for i in cust.order if i in ['coffee', 'tea']]
        esps = [i for i in cust.order if i in ['espresso', 'latte', 'cappuccino']]
        foods = [i for i in cust.order if i in ['bagel', 'sandwich', 'donut', 'wrap', 'breakfast']]
        
        self.active_kitchen_orders[cust] = {'URN': len(urns)>0, 'ESP': len(esps)>0, 'FOOD': len(foods)}
        
        if urns: 
            self.q_urn.append(cust); cust.entry_times['URN'] = self.sim_time
        if esps: 
            self.q_esp.append(cust); cust.entry_times['ESP'] = self.sim_time
        if foods: 
            for item in foods: self.q_food.append((cust, item))
            cust.entry_times['FOOD'] = self.sim_time
            
        self.check_sync(cust)
        self.try_urn(); self.try_esp(); self.try_food()

    def try_urn(self):
        avail = NUM_BARISTAS - (self.busy_urn_count + self.busy_esp_count)
        if avail <= 0 or self.urn_brewing: return
        if self.urn_cups <= 0:
            self.urn_brewing = True; self.time_next_event[8] = self.sim_time + 5.0
            return
        if self.q_urn:
            cust = self.q_urn.popleft()
            cust.queue_delays['URN'] += (self.sim_time - cust.entry_times['URN'])
            self.busy_urn_count += 1; self.urn_cups -= 1
            dur = sum(self.get_service_time(MENU_CATALOG[i]['time']) for i in cust.order if i in ['coffee', 'tea'])
            self.total_busy_time['URN'] += dur
            self.urn_finish_events.append((self.sim_time + dur, cust))

    def try_esp(self):
        if self.esp_maintenance: return 

        avail = NUM_BARISTAS - (self.busy_urn_count + self.busy_esp_count)
        if avail <= 0: return
        if self.q_esp:
            cust = self.q_esp.popleft()
            cust.queue_delays['ESP'] += (self.sim_time - cust.entry_times['ESP'])
            self.busy_esp_count += 1
            dur = sum(self.get_service_time(MENU_CATALOG[i]['time']) for i in cust.order if i in ['espresso', 'latte', 'cappuccino'])
            self.total_busy_time['ESP'] += dur
            self.esp_finish_events.append((self.sim_time + dur, cust))

    def try_food(self):
        while self.busy_cook_count < NUM_COOKS and self.q_food:
            cust, item = self.q_food.popleft()
            cust.queue_delays['FOOD'] += (self.sim_time - cust.entry_times['FOOD'])
            self.busy_cook_count += 1
            dur = self.get_service_time(MENU_CATALOG[item]['time'])
            self.total_busy_time['FOOD'] += dur
            self.cook_finish_events.append((self.sim_time + dur, cust))

    def handle_urn_done(self):
        _, cust = self.urn_finish_events.pop(0)
        self.busy_urn_count -= 1
        self.active_kitchen_orders[cust]['URN'] = False
        self.check_sync(cust); self.try_urn(); self.try_esp()

    def handle_esp_done(self):
        _, cust = self.esp_finish_events.pop(0)
        self.busy_esp_count -= 1
        self.active_kitchen_orders[cust]['ESP'] = False
        self.check_sync(cust); self.try_urn(); self.try_esp()

    def handle_food_done(self):
        _, cust = self.cook_finish_events.pop(0)
        self.busy_cook_count -= 1
        self.active_kitchen_orders[cust]['FOOD'] -= 1
        self.check_sync(cust); self.try_food()

    def handle_urn_refill(self):
        self.urn_brewing = False; self.urn_cups = URN_BATCH_SIZE
        self.time_next_event[8] = float('inf')
        self.try_urn()

    def handle_maint_start(self):
        self.esp_maintenance = True
        duration = np.random.uniform(2, 15)
        self.time_next_event[13] = self.sim_time + duration
        self.time_next_event[12] = float('inf') # Clear current start
        
        if self.maintenance_schedule:
            next_start = self.maintenance_schedule.pop(0)
            if next_start < self.sim_time: next_start = self.sim_time + 1.0 
            self.time_next_event[12] = next_start
            
        print(f"[Time {self.sim_time:.2f}] ESPRESSO MACHINE MAINTENANCE STARTED ({duration:.2f} min).")

    def handle_maint_end(self):
        self.esp_maintenance = False
        self.time_next_event[13] = float('inf')
        print(f"[Time {self.sim_time:.2f}] ESPRESSO MACHINE ONLINE.")
        self.try_esp()

    def check_sync(self, cust):
        status = self.active_kitchen_orders[cust]
        if not status['URN'] and not status['ESP'] and status['FOOD'] == 0:
            del self.active_kitchen_orders[cust]
            self.q_pack.append(cust)
            cust.entry_times['PACK'] = self.sim_time
            self.try_pack()

    #  packing trials
    def try_pack(self):
        if self.q_pack:
            if self.q_pack[0].arrival_type != DRIVE and len(self.shelf) >= SHELF_CAPACITY:
                self.shelf_max_hits += 1 
                return

        if len(self.busy_packers) < NUM_PACKERS and self.q_pack:
            cust = self.q_pack.popleft()
            cust.queue_delays['PACK'] += (self.sim_time - cust.entry_times['PACK'])
            self.busy_packers.append(cust)
            
            dur = 0.0
            for item in cust.order:
                dur += PACK_PER_ITEM
            dur = self.get_service_time(dur)
            
            self.total_busy_time['PACK'] += dur
            self.pack_finish_events.append((self.sim_time + dur, cust))

    def handle_pack_done(self):
        _, cust = self.pack_finish_events.pop(0)
        self.busy_packers.remove(cust)
        cust.kitchen_finish_time = self.sim_time
        
        if cust.arrival_type == DRIVE:
            cust.is_packed = True
            self.check_window_departure()
        else:
            self.shelf.append(cust)
            if cust.arrival_type == WALK:
                delay = self.expon(2.0)
                if self.time_next_event[7] == float('inf'): self.time_next_event[7] = self.sim_time + delay
                else: self.time_next_event[7] = min(self.time_next_event[7], self.sim_time + delay)

        self.try_pack()

    def check_window_departure(self):
        while self.dt_window_queue:
            head_cust = self.dt_window_queue[0]
            if head_cust.is_packed:
                departing = self.dt_window_queue.popleft()
                departing.window_depart_time = self.sim_time
                self.completed_list.append(departing)
                self.drive_lane_count -= 1
            else:
                break

    # Support
    def handle_mobile_check(self):
        time, cust = self.mobile_arrival_events.pop(0)
        if cust in self.shelf:
            self.shelf.remove(cust); self.completed_list.append(cust)
            self.try_pack()
        elif cust.kitchen_finish_time > 0 and cust.kitchen_finish_time <= time:
            self.completed_list.append(cust)
        else:
            LOSS_LOG.append({'time': time, 'cust': cust, 'reason': 'RENEGE (Not Ready)'})

    def handle_pickup(self):
        if self.shelf:
            cust = self.shelf.popleft()
            if cust.is_dine_in: self.try_seat_customer(cust)
            else: self.completed_list.append(cust)
        
        self.time_next_event[7] = float('inf')
        self.try_pack()
        if self.shelf: self.time_next_event[7] = self.sim_time + 1.0

    def try_seat_customer(self, cust):
        cust.entry_times['SEAT'] = self.sim_time
        if self.tables_free > 0:
            self.tables_free -= 1
            dur = self.expon(15.0)
            self.eating_finish_events.append((self.sim_time + dur, cust))
            self.total_busy_time['SEAT'] += dur
        else:
            self.q_seating.append(cust)

    def handle_eating_done(self):
        _, cust = self.eating_finish_events.pop(0)
        self.completed_list.append(cust)
        self.tables_dirty += 1
        self.try_clean()

    def try_clean(self):
        if self.tables_dirty > 0 and self.busy_busser_count < NUM_BUSSERS:
            self.tables_dirty -= 1; self.busy_busser_count += 1
            self.total_busy_time['BUS'] += CLEAN_TIME
            self.cleaning_finish_events.append((self.sim_time + CLEAN_TIME, "Table"))

    def handle_cleaning_done(self):
        self.cleaning_finish_events.pop(0)
        self.busy_busser_count -= 1; self.tables_free += 1
        if self.q_seating and self.tables_free > 0:
            next_c = self.q_seating.popleft()
            next_c.queue_delays['SEAT'] += (self.sim_time - next_c.entry_times['SEAT'])
            self.try_seat_customer(next_c)
        self.try_clean()


REPLICATION_SEEDS = [1, 2, 5, 30, 42]
CUSTOMER_SEED = 42 # Arrivals for same day
replication_results = []

print(f"\n{'='*80}")
print(f"STARTING CRN SIMULATION (Same Customers, Different Internal Events)")
print(f"Customer Seed: {CUSTOMER_SEED} | Replication Seeds: {REPLICATION_SEEDS}")
print(f"{'='*80}")

print(f">>> GENERATING WORKLOAD (Seed {CUSTOMER_SEED})...")
gen = ScenarioGenerator(CUSTOMER_SEED)
master_workload = gen.generate_workload()
print(f"    Generated {len(master_workload)} customers.")

for seed in REPLICATION_SEEDS:
    print(f"\n>>> RUNNING REPLICATION WITH INTERNAL SEED {seed}...")
    
    np.random.seed(seed) 
    LOSS_LOG.clear() 
    
    run_workload = copy.deepcopy(master_workload)

    sim = RestaurantSimulation(run_workload)
    sim.run()
    
    completed_set = set(sim.completed_list)
    walk_ins_on_shelf = [c for c in sim.shelf if c.arrival_type == WALK]
    served_customers = completed_set | set(walk_ins_on_shelf) | {e[1] for e in sim.eating_finish_events}
    
    walk_served = len([c for c in served_customers if c.arrival_type == WALK])
    drive_served = len([c for c in served_customers if c.arrival_type == DRIVE])
    mobile_served = len([c for c in served_customers if c.arrival_type == MOBILE])
    total_served = len(served_customers)

    gross_revenue = sum(c.order_value for c in served_customers)
    food_costs = sum(c.order_cost for c in served_customers)
    penalties = sum(l['cust'].order_value * 1.0 for l in LOSS_LOG) 
    renege_waste = sum(l['cust'].order_cost for l in LOSS_LOG if 'RENEGE' in l['reason'])
    
    total_staff = NUM_COOKS + NUM_BARISTAS + NUM_PACKERS + NUM_CASHIERS + NUM_BUSSERS
    labor = total_staff * HOURLY_WAGE * (CLOSE_TIME/60)
    net_profit = gross_revenue - food_costs - labor - penalties - renege_waste
    
    # Wait Times
    all_sys_times = []
    sys_times_walk = []
    sys_times_drive = []
    sys_times_mobile = []

    for c in served_customers:
        end_time = c.kitchen_finish_time
        if c.arrival_type == DRIVE and c.window_depart_time > 0:
            end_time = c.window_depart_time
        elif c.is_dine_in:
            end_time = c.kitchen_finish_time 
        
        val = end_time - c.arrival_time
        all_sys_times.append(val)
        if c.arrival_type == WALK: sys_times_walk.append(val)
        elif c.arrival_type == DRIVE: sys_times_drive.append(val)
        elif c.arrival_type == MOBILE: sys_times_mobile.append(val)

    def get_stats(times):
        if not times: return 0.0, 0.0, 0.0
        return np.mean(times), np.max(times), np.percentile(times, 95)

    w_avg, w_max, w_95 = get_stats(sys_times_walk)
    d_avg, d_max, d_95 = get_stats(sys_times_drive)
    m_avg, m_max, m_95 = get_stats(sys_times_mobile)

    balk_count = len([l for l in LOSS_LOG if 'BALK' in l['reason']])
    renege_count = len([l for l in LOSS_LOG if 'RENEGE' in l['reason']])

    # Print Report for this Seed
    print("\n" + "="*80)
    print(f"{f'FINAL STATISTICS -- SEED {seed}':^80}")
    print("="*80)
    print(f"THROUGHPUT:")
    print(f"  Walk-In:    {walk_served}")
    print(f"  Drive-Thru: {drive_served}")
    print(f"  Mobile:     {mobile_served}")
    print(f"  TOTAL:      {total_served}")
    print("-" * 80)
    print(f"WAIT TIMES (Arrival -> Food Ready/Depart) [Minutes]:")
    print(f"  {'CHANNEL':<12} {'AVG':<10} {'MAX':<10} {'95th %ILE':<15}")
    print(f"  {'Walk-In':<12} {w_avg:<10.2f} {w_max:<10.2f} {w_95:<15.2f}")
    print(f"  {'Drive-Thru':<12} {d_avg:<10.2f} {d_max:<10.2f} {d_95:<15.2f}")
    print(f"  {'Mobile':<12} {m_avg:<10.2f} {m_max:<10.2f} {m_95:<15.2f}")
    print("-" * 80)
    print(f"LOSS ANALYSIS:")
    print(f"  Drive-Thru Balks: {balk_count}")
    print(f"  Mobile Reneges:   {renege_count}")
    print(f"  Food Waste Cost:  ${renege_waste:.2f}")
    print("-" * 80)
    print(f"FINANCIALS:")
    print(f"  Gross Revenue:    ${gross_revenue:,.2f}")
    print(f"  Food Costs:       ${food_costs + renege_waste:,.2f}")
    print(f"  Labor Cost:       ${labor:,.2f} ({total_staff} Staff)")
    print(f"  Penalties:        ${penalties:,.2f}")
    print(f"  NET PROFIT:       ${net_profit:,.2f}")
    print("-" * 80)
    print(f"RESOURCE UTILIZATION:")
    print(f"  {'STATION':<10} {'BUSY TIME (m)':<15} {'CAPACITY':<10} {'UTIL %':<10}")
    
    def print_util(name, busy_time, capacity):
        total_avail = capacity * CLOSE_TIME
        util = (busy_time / total_avail) * 100
        print(f"  {name:<10} {busy_time:<15.2f} {capacity:<10} {util:<10.2f}")

    print_util("URN", sim.total_busy_time['URN'], NUM_BARISTAS)
    print_util("ESPRESSO", sim.total_busy_time['ESP'], NUM_BARISTAS)
    print_util("FOOD", sim.total_busy_time['FOOD'], NUM_COOKS)
    print_util("PACKER", sim.total_busy_time['PACK'], NUM_PACKERS)
    print_util("SEATING", sim.total_busy_time['SEAT'], NUM_TABLES)
    print_util("BUSSER", sim.total_busy_time['BUS'], NUM_BUSSERS)
    print("="*80)
    
    replication_results.append({
        'seed': seed,
        'profit': net_profit,
        'throughput': total_served,
        'avg_wait': np.mean(all_sys_times) if all_sys_times else 0,
        'tail_wait': np.percentile(all_sys_times, 95) if all_sys_times else 0,
        'balks': balk_count,
        'reneges': renege_count
    })

# Summary statistics
print("\n" + "#"*80)
print(f"{'CROSS-REPLICATION SUMMARY (5 RUNS)':^80}")
print("#"*80)

def get_ci(data):
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    h = 2.776 * std_err # t-value for 95% CI
    return mean, h

prof_m, prof_h = get_ci([r['profit'] for r in replication_results])
wait_m, wait_h = get_ci([r['avg_wait'] for r in replication_results])
tail_m, tail_h = get_ci([r['tail_wait'] for r in replication_results])
thru_m, thru_h = get_ci([r['throughput'] for r in replication_results])

print(f"{'METRIC':<20} {'MEAN':<15} {'95% CI (+/-)':<15}")
print("-" * 60)
print(f"{'Net Profit':<20} ${prof_m:<14.2f} ${prof_h:<14.2f}")
print(f"{'Throughput':<20} {thru_m:<15.1f} {thru_h:<15.1f}")
print(f"{'Avg Wait (All)':<20} {wait_m:<15.2f} {wait_h:<15.2f}")
print(f"{'95% Tail Wait':<20} {tail_m:<15.2f} {tail_h:<15.2f}")
print("-" * 60)
print(f"Avg Balks/Day:   {np.mean([r['balks'] for r in replication_results]):.1f}")
print(f"Avg Reneges/Day: {np.mean([r['reneges'] for r in replication_results]):.1f}")
print("="*80)
