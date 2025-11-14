from typing import List, Dict
from models.data_models import Benchmark, Channel

class BenchmarkRetriever:
    def __init__(self, benchmarks: List[Benchmark]):
        self.benchmarks = {benchmark.channel: benchmark for benchmark in benchmarks}
    
    def get_channel_benchmark(self, channel: Channel) -> Benchmark:
        """Get benchmark for a specific channel"""
        return self.benchmarks.get(channel)
    
    def compare_performance(self, actual_roi: float, channel: Channel) -> Dict:
        """Compare actual performance against benchmarks"""
        benchmark = self.get_channel_benchmark(channel)
        if not benchmark:
            return {}
        
        industry_gap = actual_roi - benchmark.industry_avg_roi
        top_performer_gap = actual_roi - benchmark.top_performer_roi
        
        return {
            'industry_gap': industry_gap,
            'top_performer_gap': top_performer_gap,
            'performance_ratio': actual_roi / benchmark.industry_avg_roi if benchmark.industry_avg_roi > 0 else 1.0,
            'improvement_opportunity': max(0, benchmark.top_performer_roi - actual_roi)
        }
    
    def get_recommended_budget_range(self, channel: Channel, current_spend: float) -> Dict:
        """Get recommended budget range based on benchmarks"""
        benchmark = self.get_channel_benchmark(channel)
        if not benchmark:
            return {'min': current_spend * 0.8, 'max': current_spend * 1.2}
        
        return {
            'min': max(benchmark.min_budget, current_spend * 0.5),
            'max': min(benchmark.max_budget, current_spend * 2.0),
            'optimal_range': (benchmark.min_budget * 1.2, benchmark.max_budget * 0.8)
        }