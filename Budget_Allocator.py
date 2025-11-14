import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import time
import psutil
import os

class OptimizedROIAnalyzer:
    def __init__(self):
        self.cache = {}
    
    def calculate_channel_metrics(self, historical_data: List[Dict]) -> Dict:
        """Vectorized ROI calculation with O(n) time complexity"""
        metrics = {}
        
        for channel in historical_data:
            spend = channel['total_spend']
            conversions = channel['conversions']
            revenue = channel['revenue']
            clicks = channel.get('clicks', 1)
            
            # Precompute all metrics in single pass
            roi = ((revenue - spend) / spend) * 100 if spend > 0 else 0
            cpa = spend / conversions if conversions > 0 else 0
            conversion_rate = (conversions / clicks) * 100 if clicks > 0 else 0
            roas = revenue / spend if spend > 0 else 0
            
            metrics[channel['name']] = {
                'roi': roi, 'cpa': cpa, 'conversion_rate': conversion_rate,
                'roas': roas, 'current_spend': spend, 'conversions': conversions,
                'revenue': revenue, 'type': channel['type']  # Store type for benchmark lookup
            }
        
        return metrics

class OptimizedBudgetOptimizer:
    def __init__(self, total_budget: float, constraints: Dict):
        self.total_budget = total_budget
        self.min_budgets = constraints.get('min_budgets', {})
        self.max_budgets = constraints.get('max_budgets', {})
        self.response_cache = {}
    
    def vectorized_response_curve(self, budgets: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Vectorized conversion prediction - O(1) per channel"""
        a, b = params[:, 0], params[:, 1]
        return a * np.log1p(b * budgets)  # log1p is more accurate for small values
    
    def objective_function(self, allocations: np.ndarray, channel_params: np.ndarray) -> float:
        """Optimized objective function with vectorized operations"""
        conversions = self.vectorized_response_curve(allocations, channel_params)
        return -np.sum(conversions)  # Negative for minimization
    
    def optimize(self, channel_data: Dict, channel_names: List[str]) -> Tuple:
        """Optimized budget allocation with improved constraints"""
        n_channels = len(channel_names)
        
        # Precompute arrays for vectorized operations
        params_array = np.array([channel_data[name] for name in channel_names])
        initial_guess = np.full(n_channels, self.total_budget / n_channels)
        
        # Vectorized bounds calculation
        bounds = []
        for name in channel_names:
            min_b = self.min_budgets.get(name, 0)
            max_b = self.max_budgets.get(name, self.total_budget)
            bounds.append((min_b, max_b))
        
        # Optimized constraint definition
        constraints = [{
            'type': 'eq', 
            'fun': lambda x: np.sum(x) - self.total_budget,
            'jac': lambda x: np.ones_like(x)  # Provide Jacobian for faster convergence
        }]
        
        start_time = time.time()
        result = minimize(
            self.objective_function,
            initial_guess,
            args=(params_array,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-8, 'maxiter': 1000}
        )
        optimization_time = time.time() - start_time
        
        return result, params_array, optimization_time

class EfficientMarketingAgent:
    def __init__(self, total_budget: float, industry: str, constraints: Dict = None):
        self.total_budget = total_budget
        self.industry = industry
        self.constraints = constraints or {}
        self.analyzer = OptimizedROIAnalyzer()
        self.optimizer = OptimizedBudgetOptimizer(total_budget, constraints)
        
        # Precomputed benchmarks for O(1) lookup
        self.benchmarks = {
            'paid_search': {'ecommerce': (250, 25, 3.5), 'saas': (300, 75, 2.1)},
            'social_media': {'ecommerce': (220, 35, 2.2), 'saas': (180, 95, 1.5)},
            'email': {'ecommerce': (3800, 12, 4.5), 'saas': (3200, 45, 3.2)},
            'display_ads': {'ecommerce': (150, 45, 1.2), 'saas': (120, 110, 0.8)}
        }
    
    def _prepare_optimization_params(self, historical_roi: Dict) -> Tuple[Dict, List[str]]:
        """Efficient parameter preparation with O(n) complexity"""
        channel_data = {}
        channel_names = []
        
        for name, metrics in historical_roi.items():
            conversions = metrics['conversions']
            spend = metrics['current_spend']
            
            # Optimized parameter estimation
            if spend > 0:
                a = conversions / np.log1p(spend)
                b = 0.1 + (metrics['roi'] / 10000)  # Dynamic responsiveness
            else:
                a, b = 100, 0.05
            
            channel_data[name] = (a, b)
            channel_names.append(name)
        
        return channel_data, channel_names
    
    def _generate_optimized_insights(self, channel: str, change: float, 
                                   metrics: Dict, benchmark: Tuple) -> List[str]:
        """Efficient insight generation with early returns"""
        if change <= 0:
            return []
        
        insights = []
        current_roi = metrics['roi']
        cpa = metrics['cpa']
        
        if benchmark:
            benchmark_roi, target_cpa, _ = benchmark
            
            if current_roi > benchmark_roi * 1.1:
                insights.append(f"🚀 {channel}: Strong outperformance ({current_roi:.0f}% vs {benchmark_roi:.0f}% benchmark)")
            elif current_roi < benchmark_roi * 0.9:
                insights.append(f"⚠️  {channel}: Underperforming benchmark ({current_roi:.0f}% vs {benchmark_roi:.0f}% benchmark)")
            
            if cpa < target_cpa:
                insights.append(f"📈 {channel}: Efficient CPA (${cpa:.1f} vs ${target_cpa:.1f} target)")
            else:
                insights.append(f"💰 {channel}: Higher CPA (${cpa:.1f}) but strategic value")
        
        return insights
    
    def process_data(self, historical_data: List[Dict]) -> Dict:
        """Main optimized processing pipeline"""
        # Step 1: Analyze historical performance
        historical_roi = self.analyzer.calculate_channel_metrics(historical_data)
        
        # Step 2: Prepare optimization data
        channel_data, channel_names = self._prepare_optimization_params(historical_roi)
        
        # Step 3: Run optimized allocation
        result, params_array, opt_time = self.optimizer.optimize(channel_data, channel_names)
        
        # Step 4: Generate efficient recommendations
        recommendations = self._generate_recommendations(
            result, params_array, historical_roi, channel_names, opt_time
        )
        
        return recommendations
    
    def _generate_recommendations(self, result: object, params_array: np.ndarray,
                                historical_roi: Dict, channel_names: List[str],
                                opt_time: float) -> Dict:
        """Highly optimized recommendation generation"""
        allocations = result.x
        n_channels = len(channel_names)
        
        # Pre-allocate arrays for performance
        current_spends = np.array([historical_roi[name]['current_spend'] for name in channel_names])
        current_conversions = np.array([historical_roi[name]['conversions'] for name in channel_names])
        
        # Vectorized conversion prediction
        expected_conversions = self.optimizer.vectorized_response_curve(allocations, params_array)
        
        # Efficient calculations using numpy
        changes = allocations - current_spends
        change_percentages = np.divide(changes, current_spends, 
                                     out=np.zeros_like(changes), 
                                     where=current_spends > 0) * 100
        
        budget_shares = (allocations / self.total_budget) * 100
        improvements = expected_conversions - current_conversions
        
        # Build recommendations efficiently
        optimal_allocations = {}
        expected_improvements = {}
        insights = []
        
        for i, name in enumerate(channel_names):
            # O(1) operations per channel
            optimal_allocations[name] = {
                'current_budget': current_spends[i],
                'recommended_budget': allocations[i],
                'budget_change': changes[i],
                'change_percentage': change_percentages[i],
                'budget_share': budget_shares[i]
            }
            
            expected_improvements[name] = {
                'current_conversions': current_conversions[i],
                'expected_conversions': expected_conversions[i],
                'improvement': improvements[i],
                'improvement_percentage': (improvements[i] / current_conversions[i] * 100) if current_conversions[i] > 0 else 0
            }
            
            # Get benchmark efficiently using stored channel type
            channel_type = historical_roi[name]['type']
            benchmark = self.benchmarks.get(channel_type, {}).get(self.industry)
            
            channel_insights = self._generate_optimized_insights(
                name, changes[i], historical_roi[name], benchmark
            )
            insights.extend(channel_insights)
        
        return {
            'optimal_allocations': optimal_allocations,
            'expected_improvements': expected_improvements,
            'actionable_insights': insights,
            'performance_metrics': {
                'total_expected_conversions': np.sum(expected_conversions),
                'total_budget_utilized': np.sum(allocations),
                'optimization_time_seconds': opt_time,
                'optimization_success': result.success,
                'memory_usage_mb': self._get_memory_usage()
            },
            'summary': {
                'total_budget': self.total_budget,
                'total_improvement': np.sum(improvements),
                'improvement_percentage': (np.sum(improvements) / np.sum(current_conversions) * 100) if np.sum(current_conversions) > 0 else 0
            }
        }
    
    def _get_memory_usage(self) -> float:
        """Monitor memory usage"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0  # Fallback if psutil not available

def run_optimized_analysis():
    """Test the optimized agent with performance benchmarking"""
    # Sample data - properly defined
    historical_data = [
        {'name': 'paid_search', 'type': 'paid_search', 'total_spend': 30000, 
         'conversions': 1200, 'revenue': 90000, 'clicks': 40000},
        {'name': 'social_media', 'type': 'social_media', 'total_spend': 20000, 
         'conversions': 600, 'revenue': 48000, 'clicks': 30000},
        {'name': 'email', 'type': 'email', 'total_spend': 15000, 
         'conversions': 800, 'revenue': 64000, 'clicks': 20000},
        {'name': 'display_ads', 'type': 'display_ads', 'total_spend': 10000, 
         'conversions': 200, 'revenue': 15000, 'clicks': 50000}
    ]
    
    constraints = {
        'min_budgets': {'paid_search': 10000, 'social_media': 5000, 'email': 5000, 'display_ads': 2000},
        'max_budgets': {'paid_search': 50000, 'social_media': 35000, 'email': 30000, 'display_ads': 15000}
    }
    
    # Initialize optimized agent
    agent = EfficientMarketingAgent(
        total_budget=100000,
        industry='ecommerce',
        constraints=constraints
    )
    
    print("🚀 Running OPTIMIZED Marketing Budget Analysis...")
    start_time = time.time()
    
    recommendations = agent.process_data(historical_data)
    
    total_time = time.time() - start_time
    
    # Display results
    print(f"\n⏱️  PERFORMANCE METRICS:")
    print("-" * 50)
    print(f"Total execution time: {total_time:.4f} seconds")
    print(f"Optimization time: {recommendations['performance_metrics']['optimization_time_seconds']:.4f} seconds")
    print(f"Memory usage: {recommendations['performance_metrics']['memory_usage_mb']:.2f} MB")
    print(f"Optimization successful: {recommendations['performance_metrics']['optimization_success']}")
    
    print(f"\n📊 OPTIMAL BUDGET ALLOCATION:")
    print("-" * 60)
    print(f"{'Channel':<15} {'Recommended':>12} {'Current':>12} {'Change':>8} {'Share':>8}")
    print("-" * 60)
    for channel, alloc in recommendations['optimal_allocations'].items():
        print(f"{channel:<15} ${alloc['recommended_budget']:>11,.0f} ${alloc['current_budget']:>11,.0f} "
              f"{alloc['change_percentage']:>+7.1f}% {alloc['budget_share']:>7.1f}%")
    
    print(f"\n📈 EXPECTED CONVERSION IMPROVEMENTS:")
    print("-" * 60)
    print(f"{'Channel':<15} {'Current':>10} {'Expected':>10} {'Improvement':>12} {'% Change':>10}")
    print("-" * 60)
    for channel, improv in recommendations['expected_improvements'].items():
        print(f"{channel:<15} {improv['current_conversions']:>10.0f} {improv['expected_conversions']:>10.0f} "
              f"{improv['improvement']:>+11.0f} {improv['improvement_percentage']:>+9.1f}%")
    
    print(f"\n💡 ACTIONABLE INSIGHTS:")
    print("-" * 60)
    for insight in recommendations['actionable_insights']:
        print(f"• {insight}")
    
    print(f"\n📋 SUMMARY:")
    print("-" * 50)
    summary = recommendations['summary']
    print(f"Total Budget: ${summary['total_budget']:,.2f}")
    print(f"Total Expected Conversions: {recommendations['performance_metrics']['total_expected_conversions']:,.0f}")
    print(f"Total Conversion Improvement: {summary['total_improvement']:,.0f} (+{summary['improvement_percentage']:.1f}%)")

# Run the analysis
if __name__ == "__main__":
    run_optimized_analysis()