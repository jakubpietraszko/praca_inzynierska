import { ChangeDetectionStrategy, Component, input, computed } from '@angular/core';
import { MatCardModule } from '@angular/material/card';
import { MatTableModule } from '@angular/material/table';
import { CommonModule } from '@angular/common';

export interface ClassMetric {
  class: string;
  precision: number;
  recall: number;
  f1: number;
  jaccard: number;
  roc_auc: number;
  support: number;
}

@Component({
  selector: 'app-class-metrics-table',
  imports: [MatCardModule, MatTableModule, CommonModule],
  templateUrl: './class-metrics-table.html',
  styleUrl: './class-metrics-table.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ClassMetricsTable {
  classMetrics = input.required<ClassMetric[]>();

  protected displayedColumns = computed(() => [
    'class',
    'precision',
    'recall',
    'f1',
    'jaccard',
    'roc_auc',
    'support',
  ]);
}
