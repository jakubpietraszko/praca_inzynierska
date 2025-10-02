import { ChangeDetectionStrategy, Component, input } from '@angular/core';

@Component({
  selector: 'app-content-heatmaps-statistics-image',
  imports: [],
  templateUrl: './content-heatmaps-statistics-image.html',
  styleUrl: './content-heatmaps-statistics-image.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentHeatmapsStatisticsImage {
  model = input.required<string>();
  dataset = input.required<string>();
  heatmap = input.required<string>();
  label = input.required<string>();
  image = input.required<string>();
}
