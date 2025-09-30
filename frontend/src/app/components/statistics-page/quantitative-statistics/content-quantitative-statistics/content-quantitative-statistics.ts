import { ChangeDetectionStrategy, Component, input } from '@angular/core';

@Component({
  selector: 'app-content-quantitative-statistics',
  imports: [],
  templateUrl: './content-quantitative-statistics.html',
  styleUrl: './content-quantitative-statistics.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentQuantitativeStatistics {
  model = input.required<string>();
  dataset = input.required<string>();
}
