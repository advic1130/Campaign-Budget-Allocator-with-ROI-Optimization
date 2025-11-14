from typing import List, Dict, Optional
from pydantic import BaseModel
from enum import Enum

class Channel(str, Enum):
    SEARCH = "search"
    SOCIAL = "social"
    EMAIL = "email"
    DISPLAY = "display"
    VIDEO = "video"

class ChannelROI(BaseModel):
    channel: Channel
    spend: float
    conversions: int
    revenue: float
    date: str
    
    @property
    def roi(self) -> float:
        return (self.revenue - self.spend) / self.spend if self.spend > 0 else 0
    
    @property
    def cpa(self) -> float:
        return self.spend / self.conversions if self.conversions > 0 else 0

class Benchmark(BaseModel):
    channel: Channel
    industry_avg_roi: float
    top_performer_roi: float
    min_budget: float
    max_budget: float

class OptimizationConstraints(BaseModel):
    total_budget: float
    min_channel_budget: float
    max_channel_budget: float
    required_channels: List[Channel]

class BudgetAllocation(BaseModel):
    channel: Channel
    allocated_budget: float
    expected_conversions: int
    expected_roi: float