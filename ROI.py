import numpy as np
from scipy.optimize import minimize
import pandas as pd

class ROIAnalyzer:
    def calculate_channel_roi(self, historical_data):
        """Calculate comprehensive ROI metrics by channel"""
        metrics = {}
        
        for channel in historical_data:
            spend = channel['total_spend']
            conversions = channel['conversions']
            revenue = channel['revenue']
            clicks = channel.get('clicks', 1)  # Avoid division by zero
            
            # Basic ROI calculation
            roi = ((revenue - spend) / spend) * 100 if spend > 0 else 0
            
            # Additional metrics
            metrics[channel['name']] = {
                'roi': roi,
                'cpa': spend / conversions if conversions > 0 else 0,
                'conversion_rate': (conversions / clicks) * 100 if clicks > 0 else 0,
                'roas': revenue / spend if spend > 0 else 0,
                'current_spend': spend,
                'conversions': conversions,
                'revenue': revenue
            }
        return metrics

class BenchmarkRetriever:
    def get_industry_benchmarks(self, channel_type, industry):
        """Retrieve industry-specific benchmarks"""
        benchmarks = {
            'paid_search': {
                'ecommerce': {'avg_roi': 250, 'target_cpa': 25, 'conversion_rate': 3.5},
                'saas': {'avg_roi': 300, 'target_cpa': 75, 'conversion_rate': 2.1},
                'finance': {'avg_roi': 280, 'target_cpa': 120, 'conversion_rate': 1.8}
            },
            'social_media': {
                'ecommerce': {'avg_roi': 220, 'target_cpa': 35, 'conversion_rate': 2.2},
                'saas': {'avg_roi': 180, 'target_cpa': 95, 'conversion_rate': 1.5}
            },
            'email': {
                'ecommerce': {'avg_roi': 3800, 'target_cpa': 12, 'conversion_rate': 4.5},
                'saas': {'avg_roi': 3200, 'target_cpa': 45, 'conversion_rate': 3.2}
            },
            'display_ads': {
                'ecommerce': {'avg_roi': 150, 'target_cpa': 45, 'conversion_rate': 1.2},
                'saas': {'avg_roi': 120, 'target_cpa': 110, 'conversion_rate': 0.8}
            }
        }
        return benchmarks.get(channel_type, {}).get(industry, {})

class BudgetOptimizer:
    def __init__(self, total_budget, min_channel_budgets, max_channel_budgets):
        self.total_budget = total_budget
        self.min_budgets = min_channel_budgets
        self.max_budgets = max_channel_budgets
        
    def conversion_response_curve(self, budget, channel_params):
        """Model conversion response to budget (diminishing returns)"""
        a, b = channel_params
        return a * np.log(1 + b * budget)
    
    def objective_function(self, allocations, channel_data):
        """Objective: Maximize total conversions"""
        total_conversions = 0
        channels = list(channel_data.keys())
        for i, channel in enumerate(channels):
            conversions = self.conversion_response_curve(allocations[i], channel_data[channel])
            total_conversions += conversions
        return -total_conversions  # Negative for minimization
    
    def optimize(self, channel_data):
        """Perform constrained optimization"""
        n_channels = len(channel_data)
        initial_guess = [self.total_budget / n_channels] * n_channels
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_budget}  # Total budget
        ]
        
        # Bounds for each channel
        bounds = []
        channels = list(channel_data.keys())
        for channel in channels:
            min_budget = self.min_budgets.get(channel, 0)
            max_budget = self.max_budgets.get(channel, self.total_budget)
            bounds.append((min_budget, max_budget))
        
        result = minimize(
            self.objective_function,
            initial_guess,
            args=(channel_data,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result

class MarketingOptimizationAgent:
    def __init__(self, total_budget, industry, constraints=None):
        self.total_budget = total_budget
        self.industry = industry
        self.constraints = constraints or {}
        self.analyzer = ROIAnalyzer()
        self.benchmarker = BenchmarkRetriever()
        self.optimizer = BudgetOptimizer(total_budget, 
                                       constraints.get('min_budgets', {}),
                                       constraints.get('max_budgets', {}))
    
    def _prepare_optimization_data(self, historical_roi, benchmarks):
        """Prepare channel parameters for optimization"""
        channel_data = {}
        
        for channel_name, metrics in historical_roi.items():
            # Use historical performance to estimate response curve parameters
            historical_conversions = metrics.get('conversions', 1000)
            historical_spend = metrics.get('current_spend', 10000)
            
            # Parameter estimation based on historical performance
            if historical_spend > 0:
                a = historical_conversions / np.log(1 + historical_spend)  # Scale factor
                b = 0.1  # Responsiveness factor
            else:
                a = 100  # Default scale
                b = 0.05  # Default responsiveness
            
            channel_data[channel_name] = (a, b)
        
        return channel_data
    
    def _generate_insights(self, channel, budget_change, metrics, benchmark):
        """Generate specific insights for each channel"""
        insights = []
        
        # Compare with benchmarks
        if benchmark:
            current_roi = metrics.get('roi', 0)
            benchmark_roi = benchmark.get('avg_roi', 0)
            
            if current_roi > benchmark_roi * 1.1:
                insights.append(
                    f"🚀 {channel}: Strong outperformance vs benchmark ({current_roi:.0f}% vs {benchmark_roi:.0f}%). Consider further investment."
                )
            elif current_roi < benchmark_roi * 0.9:
                insights.append(
                    f"⚠️  {channel}: Underperforming benchmark. Investigate strategy before increasing budget."
                )
        
        # Efficiency insights
        cpa = metrics.get('cpa', 0)
        target_cpa = benchmark.get('target_cpa', float('inf')) if benchmark else float('inf')
        
        if budget_change > 0 and cpa < target_cpa:
            insights.append(
                f"📈 {channel}: Efficient CPA (${cpa:.2f}). Budget increase likely to drive incremental conversions."
            )
        elif budget_change > 0 and cpa > target_cpa:
            insights.append(
                f"💰 {channel}: Higher CPA (${cpa:.2f}) but strategic value. Monitor performance closely."
            )
        
        return insights
    
    def _generate_recommendations(self, optimization_result, historical_roi, benchmarks):
        """Generate actionable budget recommendations"""
        allocations = optimization_result.x
        channels = list(historical_roi.keys())
        
        recommendations = {
            'optimal_allocations': {},
            'expected_improvements': {},
            'actionable_insights': [],
            'performance_summary': {}
        }
        
        total_expected_conversions = -optimization_result.fun
        
        # Prepare optimization data for conversion prediction
        channel_data = self._prepare_optimization_data(historical_roi, benchmarks)
        
        for i, channel in enumerate(channels):
            current_spend = historical_roi[channel].get('current_spend', 0)
            optimal_spend = allocations[i]
            change = optimal_spend - current_spend
            
            # Calculate expected conversions
            expected_conversions = self.optimizer.conversion_response_curve(
                optimal_spend, channel_data[channel]
            )
            current_conversions = historical_roi[channel].get('conversions', 0)
            
            recommendations['optimal_allocations'][channel] = {
                'current_budget': current_spend,
                'recommended_budget': optimal_spend,
                'budget_change': change,
                'change_percentage': (change / current_spend * 100) if current_spend > 0 else 100,
                'budget_share': (optimal_spend / self.total_budget) * 100
            }
            
            recommendations['expected_improvements'][channel] = {
                'current_conversions': current_conversions,
                'expected_conversions': expected_conversions,
                'improvement': expected_conversions - current_conversions,
                'improvement_percentage': ((expected_conversions - current_conversions) / current_conversions * 100) if current_conversions > 0 else 100
            }
            
            # Generate insights
            insights = self._generate_insights(
                channel, change, historical_roi[channel], benchmarks.get(channel, {})
            )
            recommendations['actionable_insights'].extend(insights)
        
        recommendations['summary'] = {
            'total_expected_conversions': total_expected_conversions,
            'total_budget_utilized': np.sum(allocations),
            'optimization_success': optimization_result.success,
            'total_budget': self.total_budget
        }
        
        return recommendations
    
    def process_data(self, historical_data):
        """Main pipeline execution"""
        # Step 1: Analyze historical performance
        historical_roi = self.analyzer.calculate_channel_roi(historical_data)
        
        # Step 2: Retrieve benchmarks
        benchmarks = {}
        for channel in historical_data:
            benchmarks[channel['name']] = self.benchmarker.get_industry_benchmarks(
                channel['type'], self.industry
            )
        
        # Step 3: Prepare optimization parameters
        channel_data = self._prepare_optimization_data(historical_roi, benchmarks)
        
        # Step 4: Run optimization
        optimization_result = self.optimizer.optimize(channel_data)
        
        # Step 5: Generate recommendations
        recommendations = self._generate_recommendations(
            optimization_result, historical_roi, benchmarks
        )
        
        return recommendations

def display_results(recommendations):
    """Display the optimization results in a formatted way"""
    print("=" * 70)
    print("🎯 MARKETING BUDGET OPTIMIZATION RESULTS")
    print("=" * 70)
    
    # Display optimal allocations
    print("\n📊 OPTIMAL BUDGET ALLOCATION:")
    print("-" * 50)
    for channel, alloc in recommendations['optimal_allocations'].items():
        print(f"{channel:15} | ${alloc['recommended_budget']:>10,.2f} | "
              f"Change: {alloc['change_percentage']:>+6.1f}% | "
              f"Share: {alloc['budget_share']:>5.1f}%")
    
    # Display conversion improvements
    print("\n📈 EXPECTED CONVERSION IMPROVEMENTS:")
    print("-" * 50)
    for channel, improv in recommendations['expected_improvements'].items():
        print(f"{channel:15} | Current: {improv['current_conversions']:>6.0f} | "
              f"Expected: {improv['expected_conversions']:>6.0f} | "
              f"Improvement: {improv['improvement']:>+6.0f}")
    
    # Display insights
    print("\n💡 ACTIONABLE INSIGHTS:")
    print("-" * 50)
    for insight in recommendations['actionable_insights']:
        print(f"• {insight}")
    
    # Display summary
    print("\n📋 SUMMARY:")
    print("-" * 50)
    summary = recommendations['summary']
    print(f"Total Budget: ${summary['total_budget']:,.2f}")
    print(f"Budget Utilized: ${summary['total_budget_utilized']:,.2f}")
    print(f"Total Expected Conversions: {summary['total_expected_conversions']:,.0f}")
    print(f"Optimization Successful: {summary['optimization_success']}")

# Sample data and execution
if __name__ == "__main__":
    # Sample historical data
    historical_data = [
        {
            'name': 'paid_search',
            'type': 'paid_search',
            'total_spend': 30000,
            'conversions': 1200,
            'revenue': 90000,
            'clicks': 40000
        },
        {
            'name': 'social_media',
            'type': 'social_media', 
            'total_spend': 20000,
            'conversions': 600,
            'revenue': 48000,
            'clicks': 30000
        },
        {
            'name': 'email',
            'type': 'email',
            'total_spend': 15000,
            'conversions': 800,
            'revenue': 64000,
            'clicks': 20000
        },
        {
            'name': 'display_ads',
            'type': 'display_ads',
            'total_spend': 10000,
            'conversions': 200,
            'revenue': 15000,
            'clicks': 50000
        }
    ]

    # Initialize agent
    agent = MarketingOptimizationAgent(
        total_budget=100000,
        industry='ecommerce',
        constraints={
            'min_budgets': {
                'paid_search': 10000, 
                'social_media': 5000, 
                'email': 5000,
                'display_ads': 2000
            },
            'max_budgets': {
                'paid_search': 50000,
                'social_media': 35000,
                'email': 30000,
                'display_ads': 15000
            }
        }
    )

    # Run optimization
    print("🚀 Running Marketing Budget Optimization...")
    recommendations = agent.process_data(historical_data)
    
    # Display results
    display_results(recommendations)