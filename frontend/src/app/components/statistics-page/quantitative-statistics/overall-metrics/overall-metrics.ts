import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { MatCardModule } from '@angular/material/card';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-overall-metrics',
  imports: [MatCardModule, CommonModule],
  templateUrl: './overall-metrics.html',
  styleUrl: './overall-metrics.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class OverallMetrics {
  accuracy = input.required<number>();
  balancedAccuracy = input.required<number>();
}
