from data.sample_data import generate_sample_roi_data, get_industry_benchmarks
from agents.data_analyzer import DataAnalyzer
from agents.benchmark_retriever import BenchmarkRetriever
from agents.optimizer import BudgetOptimizer
from models.data_models import OptimizationConstraints, Channel
import json

def main():
    print("🚀 Marketing Budget Optimization Agent")
    print("=" * 50)
    
    # Step 1: Load and analyze historical data
    print("\n📊 Step 1: Analyzing historical ROI data...")
    historical_data = generate_sample_roi_data(90)
    analyzer = DataAnalyzer(historical_data)
    performance_data = analyzer.calculate_channel_performance()
    
    print("\nHistorical Channel Performance:")
    print("-" * 40)
    for channel, perf in performance_data.items():
        print(f"{channel.value:10} | ROI: {perf['avg_roi']:.2f} | CPA: ${perf['avg_cpa']:.2f} | Trend: {perf['trend']:+.3f}")
    
    # Step 2: Retrieve and compare benchmarks
    print("\n📈 Step 2: Retrieving industry benchmarks...")
    benchmarks_data = get_industry_benchmarks()
    benchmark_retriever = BenchmarkRetriever(benchmarks_data)
    
    print("\nBenchmark Comparison:")
    print("-" * 40)
    for channel, perf in performance_data.items():
        comparison = benchmark_retriever.compare_performance(perf['avg_roi'], channel)
        if comparison:
            status = "✅ Above" if comparison['industry_gap'] >= 0 else "❌ Below"
            print(f"{channel.value:10} | {status} industry avg by {abs(comparison['industry_gap']):.2f}")
    
    # Step 3: Set up optimization constraints
    print("\n🎯 Step 3: Setting up optimization constraints...")
    constraints = OptimizationConstraints(
        total_budget=20000,
        min_channel_budget=500,
        max_channel_budget=8000,
        required_channels=[Channel.SEARCH, Channel.SOCIAL, Channel.EMAIL, Channel.DISPLAY, Channel.VIDEO]
    )
    
    print(f"Total Budget: ${constraints.total_budget:,.2f}")
    print(f"Channel Range: ${constraints.min_channel_budget:,.2f} - ${constraints.max_channel_budget:,.2f}")
    print(f"Channels: {[c.value for c in constraints.required_channels]}")
    
    # Step 4: Perform optimization
    print("\n⚡ Step 4: Performing budget optimization...")
    optimizer = BudgetOptimizer(performance_data, benchmark_retriever.benchmarks, constraints)
    
    # Calculate current performance (equal allocation for comparison)
    current_spend = {channel: constraints.total_budget / len(constraints.required_channels) 
                    for channel in constraints.required_channels}
    current_perf = optimizer.calculate_current_performance(current_spend)
    
    # Get optimized allocation
    optimized_allocations = optimizer.optimize()
    
    # Step 5: Display results
    print("\n💡 Step 5: Budget Allocation Recommendation")
    print("=" * 60)
    
    print(f"\n📈 Performance Comparison:")
    print(f"Current Allocation: {current_perf['total_conversions']:,.0f} conversions")
    
    optimized_conversions = sum(allocation.expected_conversions for allocation in optimized_allocations)
    improvement = ((optimized_conversions - current_perf['total_conversions']) / current_perf['total_conversions']) * 100
    
    print(f"Optimized Allocation: {optimized_conversions:,.0f} conversions")
    print(f"Improvement: {improvement:+.1f}%")
    
    print(f"\n💰 Recommended Budget Allocation:")
    print("-" * 50)
    print(f"{'Channel':<12} {'Budget':<12} {'% Total':<10} {'Conversions':<12} {'ROI':<8}")
    print("-" * 50)
    
    total_budget = sum(allocation.allocated_budget for allocation in optimized_allocations)
    
    for allocation in optimized_allocations:
        percentage = (allocation.allocated_budget / total_budget) * 100
        print(f"{allocation.channel.value:<12} ${allocation.allocated_budget:<11,.0f} {percentage:<9.1f}% {allocation.expected_conversions:<11} {allocation.expected_roi:<7.2f}")
    
    print("-" * 50)
    print(f"{'TOTAL':<12} ${total_budget:<11,.0f} {'100%':<9} {optimized_conversions:<11}")
    
    # Additional insights
    print(f"\n🔍 Key Insights:")
    print("-" * 30)
    
    # Find best and worst performing channels
    allocations_by_roi = sorted(optimized_allocations, key=lambda x: x.expected_roi, reverse=True)
    
    print(f" Best ROI Channel: {allocations_by_roi[0].channel.value} (ROI: {allocations_by_roi[0].expected_roi:.2f})")
    print(f"Lowest ROI Channel: {allocations_by_roi[-1].channel.value} (ROI: {allocations_by_roi[-1].expected_roi:.2f})")
    
    # Check against benchmarks
    for allocation in optimized_allocations[:2]:  # Top 2 channels
        benchmark = benchmark_retriever.get_channel_benchmark(allocation.channel)
        if benchmark and allocation.expected_roi < benchmark.industry_avg_roi:
            print(f" {allocation.channel.value} is below industry average - consider optimization")

if __name__ == "__main__":
    main()