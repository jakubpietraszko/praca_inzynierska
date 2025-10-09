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
import {
  GradientStatisticsRequest,
  GradientStatisticsResponse,
  HeatmapsStatisticsLabel,
  ShapeMetricsLabelRequest,
  ShapeMetricsLabelResponse,
} from '../heatmaps-statistics-label';
import { Subscription } from 'rxjs';
import { ImageStatisticsPlot } from '../../../shared/image-statistics-plot/image-statistics-plot';
import { HistogramPlot } from '../../../shared/histogram-plot/histogram-plot';

@Component({
  selector: 'app-content-heatmaps-statistics-label',
  imports: [ImageStatisticsPlot, HistogramPlot],
  templateUrl: './content-heatmaps-statistics-label.html',
  styleUrl: './content-heatmaps-statistics-label.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentHeatmapsStatisticsLabel implements OnDestroy {
  model = input.required<string>();
  dataset = input.required<string>();
  heatmap = input.required<string>();
  label = input.required<string>();

  heatmapsStatisticsLabelService = inject(HeatmapsStatisticsLabel);

  gradientScores = signal<GradientStatisticsResponse | null>(null);
  shapeScores = signal<ShapeMetricsLabelResponse | null>(null);

  gradientSubscription = new Subscription();
  shapeSubscription = new Subscription();

  allInputsSelected = computed(() => {
    return this.model() && this.dataset() && this.heatmap() && this.label();
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
    if (
      !scores ||
      !scores.rectangularities ||
      scores.rectangularities.length === 0
    ) {
      return [];
    }
    return scores.rectangularities;
  });

  constructor() {
    effect(() => {
      const model = this.model();
      const dataset = this.dataset();
      const heatmap = this.heatmap();
      const label = this.label();

      this.gradientSubscription.unsubscribe();
      this.gradientSubscription = new Subscription();

      this.shapeSubscription.unsubscribe();
      this.shapeSubscription = new Subscription();

      this.gradientScores.set(null);
      this.shapeScores.set(null);

      if (!model || !dataset || !heatmap || !label) {
        return;
      }

      const request: GradientStatisticsRequest = {
        model: model,
        dataset: dataset,
        heatmap: heatmap,
        percentage: 10,
        label: label,
      };

      this.gradientSubscription.add(
        this.heatmapsStatisticsLabelService.getScore(request).subscribe({
          next: (response: GradientStatisticsResponse) => {
            this.gradientScores.set(response);
          },
          error: () => {
            this.gradientScores.set(null);
          },
        }),
      );

      const shapeRequest: ShapeMetricsLabelRequest = {
        model: model,
        dataset: dataset,
        heatmap: heatmap,
        label: label,
        percentage: 15,
      };

      this.shapeSubscription.add(
        this.heatmapsStatisticsLabelService
          .getShapeScore(shapeRequest)
          .subscribe({
            next: (response: ShapeMetricsLabelResponse) => {
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
