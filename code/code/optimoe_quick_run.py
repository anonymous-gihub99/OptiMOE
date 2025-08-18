#!/usr/bin/env python3
"""
OptiMoE Quick Run Script
Streamlined version for immediate results without external dependencies
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict, deque

# Simple NetworkX replacement for core functionality
class SimpleGraph:
    """Lightweight graph implementation"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.adj = defaultdict(list)
    
    def add_node(self, node, **attrs):
        self.nodes[node] = attrs
    
    def add_edge(self, u, v, **attrs):
        self.edges[(u, v)] = attrs
        self.edges[(v, u)] = attrs
        self.adj[u].append(v)
        self.adj[v].append(u)
    
    def number_of_nodes(self):
        return len(self.nodes)
    
    def shortest_path_length(self, source, target):
        """BFS shortest path length"""
        if source == target:
            return 0
        
        visited = {source}
        queue = deque([(source, 0)])
        
        while queue:
            node, dist = queue.popleft()
            
            for neighbor in self.adj[node]:
                if neighbor == target:
                    return dist + 1
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return float('inf')  # No path found

@dataclass
class MoEConfig:
    """MoE model configuration"""
    num_experts: int = 128
    num_nodes: int = 64
    batch_size: int = 32
    sequence_length: int = 2048
    hidden_dim: int = 4096

@dataclass
class NetworkConfig:
    """Network configuration"""
    link_bandwidth: float = 100.0  # GB/s
    link_latency: float = 1.0      # microseconds
    router_latency: float = 0.5    # microseconds

class TopologyBuilder:
    """Build network topologies"""
    
    @staticmethod
    def build_fat_tree(k: int) -> SimpleGraph:
        """Build simplified fat-tree"""
        G = SimpleGraph()
        
        # For simplicity, create a 2-level fat-tree
        num_switches = k * k // 2
        num_hosts = k * k
        
        # Add switches
        for i in range(num_switches):
            G.add_node(i, type='switch')
        
        # Add hosts
        for i in range(num_hosts):
            G.add_node(num_switches + i, type='host')
        
        # Connect switches in full mesh
        for i in range(num_switches):
            for j in range(i + 1, num_switches):
                G.add_edge(i, j, bandwidth=100.0)
        
        # Connect hosts to switches
        hosts_per_switch = num_hosts // num_switches
        for host in range(num_hosts):
            switch = host // hosts_per_switch
            G.add_edge(num_switches + host, switch, bandwidth=100.0)
        
        return G
    
    @staticmethod
    def build_mesh(k: int) -> SimpleGraph:
        """Build k×k mesh"""
        G = SimpleGraph()
        
        # Add nodes
        for i in range(k * k):
            G.add_node(i, type='host')
        
        # Add edges
        for i in range(k):
            for j in range(k):
                node = i * k + j
                
                # Right neighbor
                if j < k - 1:
                    G.add_edge(node, node + 1, bandwidth=100.0)
                
                # Down neighbor
                if i < k - 1:
                    G.add_edge(node, node + k, bandwidth=100.0)
        
        return G
    
    @staticmethod
    def build_torus(k: int) -> SimpleGraph:
        """Build k×k torus"""
        G = SimpleGraph()
        
        # Add nodes
        for i in range(k * k):
            G.add_node(i, type='host')
        
        # Add edges
        for i in range(k):
            for j in range(k):
                node = i * k + j
                
                # Right neighbor (with wrap-around)
                right = i * k + ((j + 1) % k)
                G.add_edge(node, right, bandwidth=100.0)
                
                # Down neighbor (with wrap-around)
                down = ((i + 1) % k) * k + j
                G.add_edge(node, down, bandwidth=100.0)
        
        return G

class TrafficGenerator:
    """Generate MoE traffic patterns"""
    
    def __init__(self, config: MoEConfig):
        self.config = config
        self.iteration = 0
    
    def generate_traffic(self, pattern: str) -> np.ndarray:
        """Generate traffic matrix"""
        n = self.config.num_nodes
        traffic = np.zeros((n, n))
        
        if pattern == 'hotspot':
            return self._hotspot_pattern(traffic)
        elif pattern == 'uniform':
            return self._uniform_pattern(traffic)
        elif pattern == 'regional':
            return self._regional_pattern(traffic)
        else:
            # Dynamic pattern
            phase = (self.iteration // 20) % 3
            patterns = ['hotspot', 'uniform', 'regional']
            self.iteration += 1
            return self.generate_traffic(patterns[phase])
    
    def _hotspot_pattern(self, traffic: np.ndarray) -> np.ndarray:
        """Hotspot traffic pattern"""
        n = self.config.num_nodes
        hotspots = [0, 1, 2]  # First few nodes are hotspots
        
        total_messages = self.config.batch_size * self.config.sequence_length
        
        for _ in range(total_messages):
            src = np.random.randint(n)
            
            # 80% traffic to hotspots
            if np.random.random() < 0.8:
                dst = np.random.choice(hotspots)
            else:
                dst = np.random.randint(n)
            
            if src != dst:
                traffic[src, dst] += self.config.hidden_dim * 4  # bytes
        
        return traffic
    
    def _uniform_pattern(self, traffic: np.ndarray) -> np.ndarray:
        """Uniform traffic pattern"""
        n = self.config.num_nodes
        total_messages = self.config.batch_size * self.config.sequence_length
        
        for _ in range(total_messages):
            src = np.random.randint(n)
            dst = np.random.randint(n)
            
            if src != dst:
                traffic[src, dst] += self.config.hidden_dim * 4
        
        return traffic
    
    def _regional_pattern(self, traffic: np.ndarray) -> np.ndarray:
        """Regional traffic pattern"""
        n = self.config.num_nodes
        region_size = int(np.sqrt(n))
        total_messages = self.config.batch_size * self.config.sequence_length
        
        for _ in range(total_messages):
            src = np.random.randint(n)
            src_region = src // region_size
            
            # 70% chance to stay in region
            if np.random.random() < 0.7:
                region_start = src_region * region_size
                region_end = min((src_region + 1) * region_size, n)
                dst = np.random.randint(region_start, region_end)
            else:
                dst = np.random.randint(n)
            
            if src != dst:
                traffic[src, dst] += self.config.hidden_dim * 4
        
        return traffic

class NetworkSimulator:
    """Simulate network performance"""
    
    def __init__(self, topology: SimpleGraph, config: NetworkConfig):
        self.topology = topology
        self.config = config
    
    def simulate(self, traffic_matrix: np.ndarray) -> Dict:
        """Simulate network performance"""
        total_latency = 0.0
        total_messages = 0
        max_latency = 0.0
        
        n = self.topology.number_of_nodes()
        
        for src in range(min(n, traffic_matrix.shape[0])):
            for dst in range(min(n, traffic_matrix.shape[1])):
                if src != dst and traffic_matrix[src, dst] > 0:
                    # Calculate path latency
                    hops = self.topology.shortest_path_length(src, dst)
                    
                    if hops != float('inf'):
                        # Latency = propagation + transmission + queuing
                        prop_latency = hops * self.config.link_latency
                        trans_latency = traffic_matrix[src, dst] / (self.config.link_bandwidth * 1e9 / 8)
                        queue_latency = hops * 0.1  # Simplified
                        
                        latency = prop_latency + trans_latency + queue_latency
                        
                        total_latency += latency
                        max_latency = max(max_latency, latency)
                        total_messages += 1
        
        avg_latency = total_latency / max(total_messages, 1)
        
        return {
            'average_latency': avg_latency,
            'max_latency': max_latency,
            'total_messages': total_messages,
            'throughput': 0.8 - min(avg_latency / 1000, 0.5)  # Simplified throughput
        }

class OptiMoESimulation:
    """Main OptiMoE simulation"""
    
    def __init__(self):
        self.moe_config = MoEConfig()
        self.network_config = NetworkConfig()
        self.traffic_gen = TrafficGenerator(self.moe_config)
        
        # OptiMoE parameters
        self.reconfiguration_cost = 25.0
        self.amortization_window = 15
        self.performance_history = defaultdict(list)
    
    def run_baseline(self, iterations: int = 5) -> Dict:
        """Run baseline experiments"""
        print("Running baseline experiments...")
        
        topologies = ['fattree', 'mesh', 'torus']
        results = {}
        
        for topo_name in topologies:
            print(f"  Testing {topo_name}...")
            
            k = 8 if topo_name in ['mesh', 'torus'] else 4
            topology = self._build_topology(topo_name, k)
            simulator = NetworkSimulator(topology, self.network_config)
            
            topo_results = []
            
            for i in range(iterations):
                pattern = ['hotspot', 'uniform', 'regional'][i % 3]
                traffic = self.traffic_gen.generate_traffic(pattern)
                result = simulator.simulate(traffic)
                topo_results.append(result)
            
            results[topo_name] = topo_results
        
        return results
    
    def run_optimoe(self, iterations: int = 100) -> List[Dict]:
        """Run OptiMoE experiments"""
        print("Running OptiMoE experiments...")
        
        results = []
        current_topology = 'fattree'
        current_k = 4
        
        # Performance tracking
        recent_latencies = deque(maxlen=self.amortization_window)
        topology_switches = 0
        last_switch = -self.amortization_window
        
        # Build initial topology
        topology = self._build_topology(current_topology, current_k)
        simulator = NetworkSimulator(topology, self.network_config)
        
        for iteration in range(iterations):
            # Generate traffic
            traffic = self.traffic_gen.generate_traffic('dynamic')
            traffic_stats = self._analyze_traffic(traffic)
            
            # Reconfiguration decision
            reconfigure = False
            benefit = 0.0
            
            if iteration - last_switch >= self.amortization_window and recent_latencies:
                best_topo, best_k, expected_benefit = self._select_topology(traffic_stats, recent_latencies)
                
                if (best_topo != current_topology or best_k != current_k) and \
                   expected_benefit > self.reconfiguration_cost:
                    reconfigure = True
                    benefit = expected_benefit - self.reconfiguration_cost
                    current_topology = best_topo
                    current_k = best_k
                    last_switch = iteration
                    topology_switches += 1
                    
                    # Rebuild topology
                    topology = self._build_topology(current_topology, current_k)
                    simulator = NetworkSimulator(topology, self.network_config)
            
            # Simulate
            sim_result = simulator.simulate(traffic)
            recent_latencies.append(sim_result['average_latency'])
            
            # Record results
            result = {
                'iteration': iteration,
                'topology': current_topology,
                'reconfigure': reconfigure,
                'benefit': benefit,
                'latency': sim_result['average_latency'],
                'throughput': sim_result['throughput'],
                'traffic_concentration': traffic_stats['concentration'],
                'communication_volume': np.sum(traffic)
            }
            results.append(result)
            
            if iteration % 20 == 0:
                print(f"  Iteration {iteration}: {current_topology} "
                      f"(latency={sim_result['average_latency']:.1f}μs)")
        
        print(f"OptiMoE completed! Topology switches: {topology_switches}")
        return results
    
    def _build_topology(self, topo_type: str, k: int) -> SimpleGraph:
        """Build topology"""
        if topo_type == 'fattree':
            return TopologyBuilder.build_fat_tree(k)
        elif topo_type == 'mesh':
            return TopologyBuilder.build_mesh(k)
        elif topo_type == 'torus':
            return TopologyBuilder.build_torus(k)
        else:
            raise ValueError(f"Unknown topology: {topo_type}")
    
    def _analyze_traffic(self, traffic: np.ndarray) -> Dict:
        """Analyze traffic pattern"""
        non_zero = traffic[traffic > 0]
        
        if len(non_zero) == 0:
            return {'concentration': 0.0}
        
        # Gini coefficient for concentration
        sorted_traffic = np.sort(non_zero)
        n = len(sorted_traffic)
        cumsum = np.cumsum(sorted_traffic)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        return {'concentration': gini}
    
    def _select_topology(self, traffic_stats: Dict, recent_latencies: deque) -> Tuple[str, int, float]:
        """Select optimal topology"""
        current_avg = np.mean(recent_latencies)
        concentration = traffic_stats['concentration']
        
        # Simple decision rules
        if concentration > 0.6:  # High concentration -> fat-tree
            expected_latency = current_avg * 0.8
            return 'fattree', 4, current_avg - expected_latency
        elif concentration < 0.3:  # Low concentration -> torus
            expected_latency = current_avg * 0.85
            return 'torus', 8, current_avg - expected_latency
        else:  # Medium concentration -> mesh
            expected_latency = current_avg * 0.9
            return 'mesh', 8, current_avg - expected_latency

def plot_results(baseline_results: Dict, optimoe_results: List[Dict]):
    """Plot comprehensive results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Baseline comparison
    ax = axes[0, 0]
    baseline_latencies = {}
    
    for topo, results in baseline_results.items():
        latencies = [r['average_latency'] for r in results]
        baseline_latencies[topo] = latencies
    
    if baseline_latencies:
        ax.boxplot(baseline_latencies.values(), labels=baseline_latencies.keys())
        ax.set_ylabel('Average Latency (μs)')
        ax.set_title('Baseline Topology Comparison')
        ax.grid(True, alpha=0.3)
    
    # 2. OptiMoE latency evolution
    ax = axes[0, 1]
    iterations = [r['iteration'] for r in optimoe_results]
    latencies = [r['latency'] for r in optimoe_results]
    
    ax.plot(iterations, latencies, 'b-', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Latency (μs)')
    ax.set_title('OptiMoE Latency Evolution')
    ax.grid(True, alpha=0.3)
    
    # 3. Reconfiguration events
    ax = axes[1, 0]
    reconfigs = [r['reconfigure'] for r in optimoe_results]
    reconfig_points = [i for i, r in enumerate(reconfigs) if r]
    
    if reconfig_points:
        reconfig_latencies = [latencies[i] for i in reconfig_points]
        ax.scatter(reconfig_points, reconfig_latencies, 
                  color='red', s=50, alpha=0.8, label='Reconfiguration')
        ax.legend()
    
    # Also plot traffic concentration
    concentrations = [r['traffic_concentration'] for r in optimoe_results]
    ax2 = ax.twinx()
    ax2.plot(iterations, concentrations, 'g-', alpha=0.6, label='Traffic Concentration')
    ax2.set_ylabel('Concentration', color='g')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Reconfiguration Events')
    ax.set_title('Topology Reconfigurations')
    ax.grid(True, alpha=0.3)
    
    # 4. Topology usage
    ax = axes[1, 1]
    topology_counts = {}
    for r in optimoe_results:
        topo = r['topology']
        topology_counts[topo] = topology_counts.get(topo, 0) + 1
    
    if topology_counts:
        ax.bar(topology_counts.keys(), topology_counts.values())
        ax.set_xlabel('Topology')
        ax.set_ylabel('Usage Count')
        ax.set_title('Dynamic Topology Usage')
    
    plt.tight_layout()
    plt.savefig('optimoe_quick_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\nResults Summary:")
    print("=" * 50)
    
    # Calculate improvements
    baseline_avg = np.mean([np.mean([r['average_latency'] for r in results]) 
                           for results in baseline_results.values()])
    optimoe_avg = np.mean([r['latency'] for r in optimoe_results])
    improvement = (baseline_avg - optimoe_avg) / baseline_avg * 100
    
    print(f"Baseline Average Latency: {baseline_avg:.2f} μs")
    print(f"OptiMoE Average Latency:  {optimoe_avg:.2f} μs")
    print(f"Performance Improvement:  {improvement:+.1f}%")
    
    reconfig_rate = sum(1 for r in optimoe_results if r['reconfigure']) / len(optimoe_results) * 100
    print(f"Reconfiguration Rate:     {reconfig_rate:.1f}%")
    
    print(f"\nTopology Usage:")
    for topo, count in topology_counts.items():
        pct = count / len(optimoe_results) * 100
        print(f"  {topo}: {pct:.1f}%")

def main():
    """Main execution"""
    print("OptiMoE Quick Simulation")
    print("=" * 40)
    print(f"Starting simulation at {time.strftime('%H:%M:%S')}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize simulation
    sim = OptiMoESimulation()
    
    # Adjust config for quick testing
    sim.moe_config.num_nodes = 64
    sim.moe_config.num_experts = 128
    
    print(f"Configuration:")
    print(f"  Nodes: {sim.moe_config.num_nodes}")
    print(f"  Experts: {sim.moe_config.num_experts}")
    
    # Run experiments
    start_time = time.time()
    
    baseline_results = sim.run_baseline(iterations=5)
    optimoe_results = sim.run_optimoe(iterations=100)
    
    total_time = time.time() - start_time
    
    print(f"\nSimulation completed in {total_time:.1f} seconds")
    
    # Plot results
    plot_results(baseline_results, optimoe_results)
    
    # Save results
    results_data = {
        'baseline': baseline_results,
        'optimoe': optimoe_results,
        'config': {
            'num_nodes': sim.moe_config.num_nodes,
            'num_experts': sim.moe_config.num_experts,
            'reconfiguration_cost': sim.reconfiguration_cost
        }
    }
    
    with open('optimoe_quick_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("\nResults saved to:")
    print("  - optimoe_quick_results.png")
    print("  - optimoe_quick_results.json")

if __name__ == "__main__":
    main()
