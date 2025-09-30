import { ChangeDetectionStrategy, Component, input } from '@angular/core';

@Component({
  selector: 'app-content-heatmaps-statistics',
  imports: [],
  templateUrl: './content-heatmaps-statistics.html',
  styleUrl: './content-heatmaps-statistics.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentHeatmapsStatistics {
  model = input.required<string>();
  dataset = input.required<string>();
  heatmap = input.required<string>();
}
