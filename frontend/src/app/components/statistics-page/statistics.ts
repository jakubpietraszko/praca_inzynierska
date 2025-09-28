import { Component } from '@angular/core';
import { MatTabsModule } from '@angular/material/tabs';
import { QuantitativeStatistics } from './quantitative-statistics/quantitative-statistics';
import { HeatmapsStatistics } from './heatmaps-statistics/heatmaps-statistics';
import { DiffrentStatistics } from './diffrent-statistics/diffrent-statistics';

@Component({
  selector: 'app-statistics',
  imports: [MatTabsModule, QuantitativeStatistics, HeatmapsStatistics, DiffrentStatistics],
  templateUrl: './statistics.html',
  styleUrl: './statistics.css'
})
export class Statistics {

}
