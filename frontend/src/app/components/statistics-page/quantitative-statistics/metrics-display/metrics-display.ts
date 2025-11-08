import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { MatCardModule } from '@angular/material/card';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-metrics-display',
  imports: [MatCardModule, CommonModule],
  templateUrl: './metrics-display.html',
  styleUrl: './metrics-display.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class MetricsDisplay {
  readonly type = input.required<'Macro' | 'Micro' | 'Weighted'>();
  readonly precision = input.required<number>();
  readonly recall = input.required<number>();
  readonly f1 = input.required<number>();
  readonly jaccard = input.required<number>();
  readonly rocAuc = input.required<number>();
}
