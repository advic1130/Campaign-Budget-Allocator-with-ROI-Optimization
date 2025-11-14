import pandas as pd
from typing import List, Dict
from models.data_models import ChannelROI, Channel

class DataAnalyzer:
    def __init__(self, historical_data: List[ChannelROI]):
        self.historical_data = historical_data
        self.df = self._prepare_dataframe()
    
    def _prepare_dataframe(self) -> pd.DataFrame:
        """Convert historical data to DataFrame for analysis"""
        data_dicts = [item.dict() for item in self.historical_data]
        df = pd.DataFrame(data_dicts)
        df['roi_value'] = df.apply(lambda x: (x['revenue'] - x['spend']) / x['spend'] if x['spend'] > 0 else 0, axis=1)
        df['cpa'] = df.apply(lambda x: x['spend'] / x['conversions'] if x['conversions'] > 0 else 0, axis=1)
        return df
    
    def calculate_channel_performance(self) -> Dict[Channel, Dict]:
        """Calculate key performance metrics for each channel"""
        performance = {}
        
        for channel in Channel:
            channel_data = self.df[self.df['channel'] == channel.value]
            
            if len(channel_data) == 0:
                continue
                
            performance[channel] = {
                'avg_roi': channel_data['roi_value'].mean(),
                'avg_cpa': channel_data['cpa'].mean(),
                'conversion_rate': channel_data['conversions'].sum() / channel_data['spend'].sum(),
                'total_conversions': channel_data['conversions'].sum(),
                'total_spend': channel_data['spend'].sum(),
                'stability_score': 1 - (channel_data['roi_value'].std() / channel_data['roi_value'].mean() if channel_data['roi_value'].mean() != 0 else 1),
                'trend': self._calculate_trend(channel_data, 'roi_value')
            }
        
        return performance
    
    def _calculate_trend(self, channel_data: pd.DataFrame, metric: str) -> float:
        """Calculate trend of a metric over time"""
        if len(channel_data) < 2:
            return 0
        
        # Simple linear regression for trend
        channel_data = channel_data.sort_values('date')
        x = list(range(len(channel_data)))
        y = channel_data[metric].values
        
        if len(y) < 2:
            return 0
            
        trend = (y[-1] - y[0]) / len(y) if y[0] != 0 else 0
        return trend
    
    def get_roi_elasticity(self) -> Dict[Channel, float]:
        """Calculate how ROI changes with spend (elasticity)"""
        elasticity = {}
        
        for channel in Channel:
            channel_data = self.df[self.df['channel'] == channel.value]
            if len(channel_data) < 2:
                elasticity[channel] = 1.0
                continue
            
            # Simple correlation between spend and ROI
            correlation = channel_data['spend'].corr(channel_data['roi_value'])
            # Convert to elasticity (how sensitive ROI is to spend changes)
            elasticity[channel] = max(0.1, min(2.0, 1 - abs(correlation) * 0.5))
        
        return elasticity