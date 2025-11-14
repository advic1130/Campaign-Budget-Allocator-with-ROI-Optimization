import numpy as np
from scipy.optimize import minimize
from typing import List, Dict
from models.data_models import Channel, BudgetAllocation, OptimizationConstraints

class BudgetOptimizer:
    def __init__(self, performance_data: Dict, benchmarks: Dict, constraints: OptimizationConstraints):
        self.performance_data = performance_data
        self.benchmarks = benchmarks
        self.constraints = constraints
        
    def predict_conversions(self, budget: float, channel: Channel) -> float:
        """Predict conversions for a given budget on a channel"""
        if channel not in self.performance_data:
            return 0
        
        channel_perf = self.performance_data[channel]
        conversion_rate = channel_perf['conversion_rate']
        
        # Diminishing returns model
        base_conversions = budget * conversion_rate
        # Apply diminishing returns (logarithmic scaling)
        diminishing_factor = np.log1p(budget / 1000)  # log(1 + x/1000)
        
        return base_conversions * diminishing_factor
    
    def objective_function(self, budgets: List[float]) -> float:
        """Objective function to maximize (negative conversions for minimization)"""
        total_conversions = 0
        
        for i, channel in enumerate(self.constraints.required_channels):
            if i < len(budgets):
                conversions = self.predict_conversions(budgets[i], channel)
                total_conversions += conversions
        
        # We return negative because we're using minimize
        return -total_conversions
    
    def optimize(self) -> List[BudgetAllocation]:
        """Perform constrained optimization"""
        num_channels = len(self.constraints.required_channels)
        
        # Initial guess (equal distribution)
        x0 = [self.constraints.total_budget / num_channels] * num_channels
        
        # Constraints
        constraints = [
            # Total budget constraint
            {'type': 'eq', 'fun': lambda x: sum(x) - self.constraints.total_budget},
        ]
        
        # Bounds for each channel
        bounds = []
        for channel in self.constraints.required_channels:
            benchmark = self.benchmarks.get(channel)
            if benchmark:
                min_budget = max(self.constraints.min_channel_budget, benchmark.min_budget)
                max_budget = min(self.constraints.max_channel_budget, benchmark.max_budget)
            else:
                min_budget = self.constraints.min_channel_budget
                max_budget = self.constraints.max_channel_budget
            bounds.append((min_budget, max_budget))
        
        # Perform optimization
        result = minimize(
            self.objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        # Prepare results
        allocations = []
        for i, channel in enumerate(self.constraints.required_channels):
            allocated_budget = result.x[i]
            expected_conversions = self.predict_conversions(allocated_budget, channel)
            channel_perf = self.performance_data.get(channel, {})
            expected_roi = channel_perf.get('avg_roi', 2.0)
            
            allocations.append(BudgetAllocation(
                channel=channel,
                allocated_budget=allocated_budget,
                expected_conversions=int(expected_conversions),
                expected_roi=expected_roi
            ))
        
        return allocations
    
    def calculate_current_performance(self, current_spend: Dict[Channel, float]) -> Dict:
        """Calculate performance with current spend allocation"""
        total_conversions = 0
        total_spend = sum(current_spend.values())
        
        for channel, spend in current_spend.items():
            conversions = self.predict_conversions(spend, channel)
            total_conversions += conversions
        
        return {
            'total_conversions': total_conversions,
            'total_spend': total_spend,
            'average_cpa': total_spend / total_conversions if total_conversions > 0 else 0
        }