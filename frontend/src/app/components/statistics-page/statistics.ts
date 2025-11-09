import { ChangeDetectionStrategy, Component } from '@angular/core';
import { MatTabsModule } from '@angular/material/tabs';
import { WrapperQuantitativeStatistics } from './quantitative-statistics/wrapper-quantitative-statistics/wrapper-quantitative-statistics';
import { WrapperHeatmapsStatisticsImage } from './heatmaps-statistics-image/wrapper-heatmaps-statistics-image/wrapper-heatmaps-statistics-image';
import { WrapperHeatmapsStatisticsDataset } from './heatmaps-statistics-dataset/wrapper-heatmaps-statistics-dataset/wrapper-heatmaps-statistics-dataset';
import { WrapperHeatmapsStatisticsLabel } from './heatmaps-statistics-label/wrapper-heatmaps-statistics-label/wrapper-heatmaps-statistics-label';

@Component({
  selector: 'app-statistics',
  imports: [
    MatTabsModule,
    WrapperQuantitativeStatistics,
    WrapperHeatmapsStatisticsImage,
    WrapperHeatmapsStatisticsDataset,
    WrapperHeatmapsStatisticsLabel,
  ],
  templateUrl: './statistics.html',
  styleUrl: './statistics.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Statistics {}
