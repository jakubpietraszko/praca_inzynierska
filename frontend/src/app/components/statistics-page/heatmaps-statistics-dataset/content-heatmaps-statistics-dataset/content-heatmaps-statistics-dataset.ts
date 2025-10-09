import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  input,
  OnDestroy,
  signal,
} from '@angular/core';
import { ImageStatisticsPlot } from '../../../shared/image-statistics-plot/image-statistics-plot';
import {
  GradientStatisticsRequest,
  GradientStatisticsResponse,
  HeatmapsStatisticsDataset,
} from '../heatmaps-statistics-dataset';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-content-heatmaps-statistics-dataset',
  imports: [ImageStatisticsPlot],
  templateUrl: './content-heatmaps-statistics-dataset.html',
  styleUrl: './content-heatmaps-statistics-dataset.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentHeatmapsStatisticsDataset implements OnDestroy {
  model = input.required<string>();
  dataset = input.required<string>();
  heatmap = input.required<string>();

  heatmapsStatisticsDatasetService = inject(HeatmapsStatisticsDataset);

  gradientScores = signal<GradientStatisticsResponse | null>(null);

  gradientSubscription = new Subscription();

  allInputsSelected = computed(() => {
    return this.model() && this.dataset() && this.heatmap();
  });

  averageDropXData = computed(() => {
    const scores = this.gradientScores();
    if (
      !scores ||
      !scores.list_of_average_drop ||
      scores.list_of_average_drop.length === 0
    ) {
      return [];
    }
    const scoresArray = scores.list_of_average_drop;
    if (scoresArray.length === 0) {
      return [];
    }
    if (scoresArray.length === 1) {
      return [0];
    }
    const step = 100 / (scoresArray.length - 1);
    return Array.from({ length: scoresArray.length }, (_, i) => i * step);
  });

  averageDropYData = computed(() => {
    const scores = this.gradientScores();
    if (
      !scores ||
      !scores.list_of_average_drop ||
      scores.list_of_average_drop.length === 0
    ) {
      return [];
    }
    return scores.list_of_average_drop;
  });

  rateOfIncreaseXData = computed(() => {
    const scores = this.gradientScores();
    if (
      !scores ||
      !scores.list_of_rate_of_increase_in_scores ||
      scores.list_of_rate_of_increase_in_scores.length === 0
    ) {
      return [];
    }
    const scoresArray = scores.list_of_rate_of_increase_in_scores;
    if (scoresArray.length === 0) {
      return [];
    }
    if (scoresArray.length === 1) {
      return [0];
    }
    const step = 100 / (scoresArray.length - 1);
    return Array.from({ length: scoresArray.length }, (_, i) => i * step);
  });

  rateOfIncreaseYData = computed(() => {
    const scores = this.gradientScores();
    if (
      !scores ||
      !scores.list_of_rate_of_increase_in_scores ||
      scores.list_of_rate_of_increase_in_scores.length === 0
    ) {
      return [];
    }
    return scores.list_of_rate_of_increase_in_scores;
  });

  constructor() {
    effect(() => {
      const model = this.model();
      const dataset = this.dataset();
      const heatmap = this.heatmap();

      this.gradientSubscription.unsubscribe();
      this.gradientSubscription = new Subscription();

      this.gradientScores.set(null);

      if (!model || !dataset || !heatmap) {
        return;
      }

      const request: GradientStatisticsRequest = {
        model: model,
        dataset: dataset,
        heatmap: heatmap,
        percentage: 5,
      };

      this.gradientSubscription.add(
        this.heatmapsStatisticsDatasetService.getScore(request).subscribe({
          next: (response: GradientStatisticsResponse) => {
            this.gradientScores.set(response);
          },
          error: () => {
            this.gradientScores.set(null);
          },
        }),
      );
    });
  }

  ngOnDestroy(): void {
    this.gradientSubscription.unsubscribe();
  }
}
