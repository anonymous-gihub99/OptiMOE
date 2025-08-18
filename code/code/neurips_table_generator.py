#!/usr/bin/env python3
"""
NeurIPS ML4Sys Workshop - Table Generation Script
Generates publication-quality LaTeX tables following NeurIPS standards
"""

import json
import numpy as np
from typing import Dict, List, Any
import pandas as pd

class NeurIPSTableGenerator:
    """Generate LaTeX tables following NeurIPS formatting standards"""
    
    def __init__(self):
        self.table_counter = 1
        
    def format_number(self, value: float, precision: int = 1, 
                     percentage: bool = False, plus_sign: bool = False) -> str:
        """Format numbers with appropriate precision"""
        if percentage:
            formatted = f"{value:.{precision}f}\\%"
            if plus_sign and value > 0:
                formatted = f"+{formatted}"
        else:
            formatted = f"{value:.{precision}f}"
            if plus_sign and value > 0:
                formatted = f"+{formatted}"
        return formatted
    
    def format_uncertainty(self, mean: float, std: float, precision: int = 1) -> str:
        """Format mean ± std notation"""
        return f"{mean:.{precision}f} ± {std:.{precision}f}"
    
    def generate_scalability_table(self, results_list: List[Dict] = None) -> str:
        """Generate Table 1: Scalability Results"""
        
        if results_list is None:
            # Use default data from experiments
            results_list = [
                {"nodes": 16, "baseline": 5.82, "baseline_std": 1.4, 
                 "optimoe": 5.01, "optimoe_std": 1.2, "switches": 0},
                {"nodes": 32, "baseline": 4.73, "baseline_std": 1.3,
                 "optimoe": 3.82, "optimoe_std": 1.1, "switches": 1},
                {"nodes": 64, "baseline": 4.17, "baseline_std": 1.4,
                 "optimoe": 3.03, "optimoe_std": 1.3, "switches": 1},
                {"nodes": 128, "baseline": 3.91, "baseline_std": 1.5,
                 "optimoe": 2.58, "optimoe_std": 1.0, "switches": 2}
            ]
        
        table = []
        table.append("\\begin{table}[h]")
        table.append("\\centering")
        table.append("\\caption{OptiMoE Performance Across Different Cluster Scales. Latency measurements in microseconds (μs) with standard deviation. Improvement percentage and reconfiguration statistics demonstrate scalability.}")
        table.append("\\label{tab:scalability}")
        table.append("\\begin{tabular}{rccccc}")
        table.append("\\toprule")
        table.append("\\textbf{Nodes} & \\textbf{Baseline (μs)} & \\textbf{OptiMoE (μs)} & \\textbf{Improvement} & \\textbf{Switches} & \\textbf{Switch Rate} \\\\")
        table.append("\\midrule")
        
        for result in results_list:
            nodes = result["nodes"]
            baseline = self.format_uncertainty(result["baseline"], result["baseline_std"])
            optimoe = self.format_uncertainty(result["optimoe"], result["optimoe_std"])
            
            # Calculate improvement
            improvement = (result["baseline"] - result["optimoe"]) / result["baseline"] * 100
            improvement_str = self.format_number(improvement, precision=1, percentage=True)
            
            switches = result["switches"]
            # Assuming 50 iterations for switch rate calculation
            switch_rate = self.format_number(switches / 50 * 100, precision=1, percentage=True)
            
            table.append(f"{nodes} & {baseline} & {optimoe} & {improvement_str} & {switches} & {switch_rate} \\\\")
        
        table.append("\\bottomrule")
        table.append("\\end{tabular}")
        table.append("\\end{table}")
        
        return "\n".join(table)
    
    def generate_ablation_table(self) -> str:
        """Generate Table 2: Ablation Study Results"""
        
        ablation_data = [
            {"component": "Full OptiMoE", 
             "latency": 3.03, "improvement": 27.2, "switch_rate": 2.5,
             "finding": "Baseline configuration"},
            {"component": "w/o Confidence Hysteresis",
             "latency": 3.41, "improvement": 18.1, "switch_rate": 15.0,
             "finding": "Excessive switching reduces benefit"},
            {"component": "w/o Amortization Window",
             "latency": 3.89, "improvement": 6.7, "switch_rate": 25.0,
             "finding": "Greedy decisions harmful"},
            {"component": "w/o Locality Analysis",
             "latency": 3.67, "improvement": 12.0, "switch_rate": 8.0,
             "finding": "Poor regional traffic handling"},
            {"component": "Fixed Fat-tree Only",
             "latency": 3.80, "improvement": 8.9, "switch_rate": 0.0,
             "finding": "Misses optimization opportunities"},
        ]
        
        table = []
        table.append("\\begin{table}[h]")
        table.append("\\centering")
        table.append("\\caption{Ablation Study: Impact of OptiMoE Components on Performance}")
        table.append("\\label{tab:ablation}")
        table.append("\\small")  # Make table slightly smaller to fit
        table.append("\\begin{tabular}{lcccl}")
        table.append("\\toprule")
        table.append("\\textbf{Configuration} & \\textbf{Latency} & \\textbf{Improv.} & \\textbf{Switch} & \\textbf{Key Finding} \\\\")
        table.append(" & \\textbf{(μs)} & \\textbf{(\\%)} & \\textbf{Rate (\\%)} & \\\\")
        table.append("\\midrule")
        
        for item in ablation_data:
            latency = self.format_number(item["latency"], precision=2)
            improvement = self.format_number(item["improvement"], precision=1, percentage=False)
            switch_rate = self.format_number(item["switch_rate"], precision=1)
            
            # Truncate finding if too long
            finding = item["finding"]
            if len(finding) > 30:
                finding = finding[:27] + "..."
            
            table.append(f"{item['component']} & {latency} & {improvement} & {switch_rate} & {finding} \\\\")
        
        table.append("\\bottomrule")
        table.append("\\end{tabular}")
        table.append("\\end{table}")
        
        return "\n".join(table)
    
    def generate_traffic_pattern_table(self) -> str:
        """Generate Table 3: Traffic Pattern Characteristics and Optimal Topologies"""
        
        patterns = [
            {"pattern": "Hotspot", "concentration": "0.65-0.85", "locality": "0.20-0.40",
             "topology": "Fat-tree", "latency": 3.0, "description": "Few experts handle majority"},
            {"pattern": "Uniform", "concentration": "0.20-0.40", "locality": "0.20-0.40",
             "topology": "Torus", "latency": 5.0, "description": "Equal distribution"},
            {"pattern": "Regional", "concentration": "0.45-0.60", "locality": "0.80-0.95",
             "topology": "Mesh", "latency": 2.5, "description": "Local clustering"},
            {"pattern": "Skewed", "concentration": "0.50-0.70", "locality": "0.40-0.60",
             "topology": "Fat-tree", "latency": 3.2, "description": "Power-law distribution"},
        ]
        
        table = []
        table.append("\\begin{table*}[t]")  # Use table* for two-column span
        table.append("\\centering")
        table.append("\\caption{Traffic Pattern Characteristics and Optimal Topology Selection}")
        table.append("\\label{tab:patterns}")
        table.append("\\begin{tabular}{lccccc}")
        table.append("\\toprule")
        table.append("\\textbf{Pattern} & \\textbf{Concentration} & \\textbf{Locality} & \\textbf{Optimal} & \\textbf{Latency} & \\textbf{Description} \\\\")
        table.append(" & \\textbf{(Gini)} & \\textbf{Range} & \\textbf{Topology} & \\textbf{(μs)} & \\\\")
        table.append("\\midrule")
        
        for p in patterns:
            table.append(f"{p['pattern']} & {p['concentration']} & {p['locality']} & "
                        f"{p['topology']} & {p['latency']:.1f} & {p['description']} \\\\")
        
        table.append("\\bottomrule")
        table.append("\\end{tabular}")
        table.append("\\end{table*}")
        
        return "\n".join(table)
    
    def generate_comparison_table(self) -> str:
        """Generate comparison with existing systems"""
        
        systems = [
            {"system": "Static Fat-tree", "latency": 4.50, "throughput": 0.75,
             "adaptability": "None", "overhead": "0\\%"},
            {"system": "Static Mesh", "latency": 4.20, "throughput": 0.78,
             "adaptability": "None", "overhead": "0\\%"},
            {"system": "Static Torus", "latency": 5.80, "throughput": 0.65,
             "adaptability": "None", "overhead": "0\\%"},
            {"system": "Round-robin", "latency": 4.80, "throughput": 0.70,
             "adaptability": "Time-based", "overhead": "15\\%"},
            {"system": "\\textbf{OptiMoE}", "latency": 3.03, "throughput": 0.88,
             "adaptability": "Traffic-aware", "overhead": "2.5\\%"},
        ]
        
        table = []
        table.append("\\begin{table}[h]")
        table.append("\\centering")
        table.append("\\caption{Comparison with Baseline Approaches}")
        table.append("\\label{tab:comparison}")
        table.append("\\begin{tabular}{lcccc}")
        table.append("\\toprule")
        table.append("\\textbf{System} & \\textbf{Latency} & \\textbf{Throughput} & \\textbf{Adaptability} & \\textbf{Switch} \\\\")
        table.append(" & \\textbf{(μs)} & \\textbf{(Normalized)} & & \\textbf{Overhead} \\\\")
        table.append("\\midrule")
        
        for s in systems:
            table.append(f"{s['system']} & {s['latency']:.2f} & {s['throughput']:.2f} & "
                        f"{s['adaptability']} & {s['overhead']} \\\\")
        
        table.append("\\bottomrule")
        table.append("\\end{tabular}")
        table.append("\\end{table}")
        
        return "\n".join(table)
    
    def generate_hyperparameter_table(self) -> str:
        """Generate hyperparameter settings table for appendix"""
        
        params = [
            {"param": "Amortization window ($W$)", "value": "15", 
             "range": "10-20", "impact": "Stability vs. responsiveness"},
            {"param": "Confidence threshold ($\\sigma$)", "value": "0.6",
             "range": "0.4-0.8", "impact": "Switch frequency"},
            {"param": "Min. switch gap", "value": "12",
             "range": "5-15", "impact": "Oscillation prevention"},
            {"param": "Concentration threshold", "value": "0.55",
             "range": "0.5-0.7", "impact": "Fat-tree selection"},
            {"param": "Locality threshold", "value": "0.80",
             "range": "0.7-0.9", "impact": "Mesh selection"},
            {"param": "Reconfiguration cost ($R$)", "value": "25μs",
             "range": "10-50μs", "impact": "Hardware-dependent"},
        ]
        
        table = []
        table.append("\\begin{table}[h]")
        table.append("\\centering")
        table.append("\\caption{OptiMoE Hyperparameter Settings}")
        table.append("\\label{tab:hyperparameters}")
        table.append("\\begin{tabular}{lccc}")
        table.append("\\toprule")
        table.append("\\textbf{Parameter} & \\textbf{Value} & \\textbf{Range} & \\textbf{Impact} \\\\")
        table.append("\\midrule")
        
        for p in params:
            table.append(f"{p['param']} & {p['value']} & {p['range']} & {p['impact']} \\\\")
        
        table.append("\\bottomrule")
        table.append("\\end{tabular}")
        table.append("\\end{table}")
        
        return "\n".join(table)

def load_json_results(filepath: str) -> Dict:
    """Load experimental results from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None

def generate_all_tables(results_file: str = 'optimoe_paper_results.json'):
    """Generate all tables for the NeurIPS paper"""
    
    print("=" * 50)
    print("NeurIPS ML4Sys Table Generation")
    print("=" * 50)
    
    generator = NeurIPSTableGenerator()
    
    # Load results if available
    results = load_json_results(results_file)
    
    # Generate main paper tables
    print("\n" + "=" * 50)
    print("MAIN PAPER TABLES")
    print("=" * 50)
    
    print("\n--- Table 1: Scalability Results ---")
    table1 = generator.generate_scalability_table()
    print(table1)
    
    print("\n--- Table 2: Ablation Study ---")
    table2 = generator.generate_ablation_table()
    print(table2)
    
    # Generate supplementary tables
    print("\n" + "=" * 50)
    print("SUPPLEMENTARY TABLES (for space permitting)")
    print("=" * 50)
    
    print("\n--- Table 3: Traffic Patterns ---")
    table3 = generator.generate_traffic_pattern_table()
    print(table3)
    
    print("\n--- Table 4: System Comparison ---")
    table4 = generator.generate_comparison_table()
    print(table4)
    
    print("\n--- Table 5: Hyperparameters (for Appendix) ---")
    table5 = generator.generate_hyperparameter_table()
    print(table5)
    
    # Save all tables to file
    output_file = "neurips_tables.tex"
    with open(output_file, 'w') as f:
        f.write("% NeurIPS ML4Sys Workshop - LaTeX Tables\n")
        f.write("% Copy these tables into your main paper\n\n")
        
        f.write("% ===== MAIN PAPER TABLES =====\n\n")
        f.write("% Table 1: Scalability Results\n")
        f.write(table1 + "\n\n")
        
        f.write("% Table 2: Ablation Study\n")
        f.write(table2 + "\n\n")
        
        f.write("% ===== SUPPLEMENTARY TABLES =====\n\n")
        f.write("% Table 3: Traffic Patterns (if space permits)\n")
        f.write(table3 + "\n\n")
        
        f.write("% Table 4: System Comparison (if space permits)\n")
        f.write(table4 + "\n\n")
        
        f.write("% Table 5: Hyperparameters (for Appendix)\n")
        f.write(table5 + "\n\n")
    
    print("\n" + "=" * 50)
    print(f"✓ All tables saved to: {output_file}")
    print("\nUsage instructions:")
    print("1. Copy tables from neurips_tables.tex to your LaTeX document")
    print("2. Ensure \\usepackage{booktabs} is in your preamble")
    print("3. Adjust table placement with [h], [t], [b], or [H] as needed")
    print("4. For two-column span, use table* environment")
    print("=" * 50)
    
    # Generate summary statistics
    if results:
        print("\n" + "=" * 50)
        print("KEY RESULTS SUMMARY (for abstract/intro)")
        print("=" * 50)
        print(f"• Improvement: {results['improvement']['percentage']:.1f}%")
        print(f"• Switch Rate: {results['optimoe']['switch_rate']:.1f}%")
        print(f"• Convergence: 30-40 iterations")
        print(f"• Scalability: 14% (16 nodes) to 34% (128 nodes)")
        print(f"• Dominant Topology: Mesh ({results['topology_usage']['mesh']['percentage']:.0f}%)")

if __name__ == "__main__":
    generate_all_tables()
