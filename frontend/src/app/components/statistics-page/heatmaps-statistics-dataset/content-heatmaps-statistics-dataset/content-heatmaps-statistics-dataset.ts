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
import { HistogramPlot } from '../../../shared/histogram-plot/histogram-plot';
import {
  GradientStatisticsRequest,
  GradientStatisticsResponse,
  HeatmapsStatisticsDataset,
  ShapeMetricsDatasetRequest,
  ShapeMetricsDatasetResponse,
} from '../heatmaps-statistics-dataset';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-content-heatmaps-statistics-dataset',
  imports: [ImageStatisticsPlot, HistogramPlot],
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
  shapeScores = signal<ShapeMetricsDatasetResponse | null>(null);

  gradientSubscription = new Subscription();
  shapeSubscription = new Subscription();

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

  areasData = computed(() => {
    const scores = this.shapeScores();
    if (!scores || !scores.areas || scores.areas.length === 0) {
      return [];
    }
    return scores.areas;
  });

  perimetersData = computed(() => {
    const scores = this.shapeScores();
    if (!scores || !scores.perimeters || scores.perimeters.length === 0) {
      return [];
    }
    return scores.perimeters;
  });

  circularitiesData = computed(() => {
    const scores = this.shapeScores();
    if (!scores || !scores.circularities || scores.circularities.length === 0) {
      return [];
    }
    return scores.circularities;
  });

  rectangularitiesData = computed(() => {
    const scores = this.shapeScores();
    if (!scores || !scores.rectangularities || scores.rectangularities.length === 0) {
      return [];
    }
    return scores.rectangularities;
  });

  constructor() {
    effect(() => {
      const model = this.model();
      const dataset = this.dataset();
      const heatmap = this.heatmap();

      this.gradientSubscription.unsubscribe();
      this.gradientSubscription = new Subscription();

      this.shapeSubscription.unsubscribe();
      this.shapeSubscription = new Subscription();

      this.gradientScores.set(null);
      this.shapeScores.set(null);

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

      const shapeRequest: ShapeMetricsDatasetRequest = {
        model: model,
        dataset: dataset,
        heatmap: heatmap,
        percentage: 15,
      };

      this.shapeSubscription.add(
        this.heatmapsStatisticsDatasetService
          .getShapeScore(shapeRequest)
          .subscribe({
            next: (response: ShapeMetricsDatasetResponse) => {
              this.shapeScores.set(response);
            },
            error: () => {
              this.shapeScores.set(null);
            },
          }),
      );
    });
  }

  ngOnDestroy(): void {
    this.gradientSubscription.unsubscribe();
    this.shapeSubscription.unsubscribe();
  }
}
