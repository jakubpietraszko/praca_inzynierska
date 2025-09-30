import { Component } from '@angular/core';
import { MatTabsModule } from '@angular/material/tabs';
import { WrapperQuantitativeStatistics } from './quantitative-statistics/wrapper-quantitative-statistics/wrapper-quantitative-statistics';
import { WrapperDifferentStatistics } from './different-statistics/wrapper-different-statistics/wrapper-different-statistics';
import { WrapperHeatmapsStatistics } from './heatmaps-statistics/wrapper-heatmaps-statistics/wrapper-heatmaps-statistics';

@Component({
  selector: 'app-statistics',
  imports: [
    MatTabsModule,
    WrapperQuantitativeStatistics,
    WrapperDifferentStatistics,
    WrapperHeatmapsStatistics,
  ],
  templateUrl: './statistics.html',
  styleUrl: './statistics.css',
})
export class Statistics {}
