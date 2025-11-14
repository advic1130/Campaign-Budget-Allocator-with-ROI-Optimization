from datetime import datetime, timedelta
from typing import List
from models.data_models import ChannelROI, Channel, Benchmark
import random

def generate_sample_roi_data(days: int = 90) -> List[ChannelROI]:
    """Generate sample historical ROI data"""
    data = []
    base_date = datetime.now() - timedelta(days=days)
    
    # Base performance characteristics for each channel
    channel_performance = {
        Channel.SEARCH: {"conversion_rate": 0.08, "avg_order_value": 150},
        Channel.SOCIAL: {"conversion_rate": 0.05, "avg_order_value": 120},
        Channel.EMAIL: {"conversion_rate": 0.12, "avg_order_value": 200},
        Channel.DISPLAY: {"conversion_rate": 0.03, "avg_order_value": 100},
        Channel.VIDEO: {"conversion_rate": 0.06, "avg_order_value": 180},
    }
    
    for i in range(days):
        current_date = base_date + timedelta(days=i)
        for channel in Channel:
            performance = channel_performance[channel]
            
            # Add some randomness to simulate real data
            daily_spend = random.uniform(100, 1000)
            noise = random.uniform(0.8, 1.2)
            
            expected_conversions = int(daily_spend * performance["conversion_rate"] * noise)
            revenue = expected_conversions * performance["avg_order_value"] * random.uniform(0.9, 1.1)
            
            data.append(ChannelROI(
                channel=channel,
                spend=daily_spend,
                conversions=max(1, expected_conversions),
                revenue=revenue,
                date=current_date.strftime("%Y-%m-%d")
            ))
    
    return data

def get_industry_benchmarks() -> List[Benchmark]:
    """Get industry benchmark data"""
    return [
        Benchmark(
            channel=Channel.SEARCH,
            industry_avg_roi=3.2,
            top_performer_roi=5.8,
            min_budget=500,
            max_budget=10000
        ),
        Benchmark(
            channel=Channel.SOCIAL,
            industry_avg_roi=2.1,
            top_performer_roi=4.2,
            min_budget=300,
            max_budget=8000
        ),
        Benchmark(
            channel=Channel.EMAIL,
            industry_avg_roi=4.5,
            top_performer_roi=7.2,
            min_budget=200,
            max_budget=5000
        ),
        Benchmark(
            channel=Channel.DISPLAY,
            industry_avg_roi=1.5,
            top_performer_roi=3.1,
            min_budget=400,
            max_budget=6000
        ),
        Benchmark(
            channel=Channel.VIDEO,
            industry_avg_roi=2.8,
            top_performer_roi=4.9,
            min_budget=600,
            max_budget=12000
        ),
    ]