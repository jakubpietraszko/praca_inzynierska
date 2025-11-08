import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  input,
} from '@angular/core';
import { GeneratorStatistics } from '../../generator-statistics';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { CommonModule } from '@angular/common';
import { ConfusionMatrixComponent } from '../confusion-matrix/confusion-matrix';
import { ModelDatasetStatisticsRequest } from '../../../../generated-interfaces/model_dataset_statistics';
import { OverallMetrics } from '../overall-metrics/overall-metrics';
import { MetricsDisplay } from '../metrics-display/metrics-display';
import { ClassMetricsTable } from '../class-metrics-table/class-metrics-table';

@Component({
  selector: 'app-content-quantitative-statistics',
  imports: [
    MatProgressSpinnerModule,
    CommonModule,
    ConfusionMatrixComponent,
    OverallMetrics,
    MetricsDisplay,
    ClassMetricsTable,
  ],
  providers: [GeneratorStatistics],
  templateUrl: './content-quantitative-statistics.html',
  styleUrl: './content-quantitative-statistics.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentQuantitativeStatistics {
  private readonly statisticsService = inject(GeneratorStatistics);

  model = input.required<string>();
  dataset = input.required<string>();

  protected readonly statisticsResponse = this.statisticsService.response;

  constructor() {
    effect(() => {
      const request: ModelDatasetStatisticsRequest = {
        dataset: this.dataset(),
        model: this.model(),
      };

      this.statisticsService.fetchStatistics(request);
    });
  }

  classNames = computed(() => {
    const response = this.statisticsResponse();
    if (!response?.confusion_matrix) return [];
    return Object.keys(response.confusion_matrix);
  });

  classMetrics = computed(() => {
    const response = this.statisticsResponse();
    if (!response) return [];

    const classNames = this.classNames();

    return classNames.map((className: string) => ({
      class: className,
      precision: response.precision_per_class[className] || 0,
      recall: response.recall_per_class[className] || 0,
      f1: response.f1_score_per_class[className] || 0,
      jaccard: response.jaccard_per_class[className] || 0,
      roc_auc: response.roc_auc_per_class[className] || 0,
      support: response.support_per_class[className] || 0,
    }));
  });

  confusionMatrix = computed(() => {
    const response = this.statisticsResponse();
    return response?.confusion_matrix || null;
  });

  accuracy = computed(() => this.statisticsResponse()?.accuracy || 0);
  balancedAccuracy = computed(
    () => this.statisticsResponse()?.balanced_accuracy || 0,
  );

  precisionMacro = computed(
    () => this.statisticsResponse()?.precision_macro || 0,
  );
  precisionMicro = computed(
    () => this.statisticsResponse()?.precision_micro || 0,
  );
  precisionWeighted = computed(
    () => this.statisticsResponse()?.precision_weighted || 0,
  );

  recallMacro = computed(() => this.statisticsResponse()?.recall_macro || 0);
  recallMicro = computed(() => this.statisticsResponse()?.recall_micro || 0);
  recallWeighted = computed(
    () => this.statisticsResponse()?.recall_weighted || 0,
  );

  f1Macro = computed(() => this.statisticsResponse()?.f1_score_macro || 0);
  f1Micro = computed(() => this.statisticsResponse()?.f1_score_micro || 0);
  f1Weighted = computed(
    () => this.statisticsResponse()?.f1_score_weighted || 0,
  );

  jaccardMacro = computed(() => this.statisticsResponse()?.jaccard_macro || 0);
  jaccardMicro = computed(() => this.statisticsResponse()?.jaccard_micro || 0);
  jaccardWeighted = computed(
    () => this.statisticsResponse()?.jaccard_weighted || 0,
  );

  rocAucMacro = computed(() => this.statisticsResponse()?.roc_auc_macro || 0);
  rocAucMicro = computed(() => this.statisticsResponse()?.roc_auc_micro || 0);
  rocAucWeighted = computed(
    () => this.statisticsResponse()?.roc_auc_weighted || 0,
  );
}
