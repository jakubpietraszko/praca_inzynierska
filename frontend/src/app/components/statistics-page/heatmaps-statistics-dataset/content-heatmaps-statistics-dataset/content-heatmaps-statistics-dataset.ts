import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  input,
} from '@angular/core';
import {
  HeatmapStatisticsDatasetManyRequest,
} from '../../../../generated-interfaces/heatmap';
import {
  ShapeMetricsDatasetRequest,
} from '../../../../generated-interfaces/shapes';
import { ImageStatisticsPlot } from '../../../shared/image-statistics-plot/image-statistics-plot';
import { HistogramPlot } from '../../../shared/histogram-plot/histogram-plot';
import { Shapes } from '../shapes';
import { Scores } from '../scores';

@Component({
  selector: 'app-content-heatmaps-statistics-dataset',
  imports: [ImageStatisticsPlot, HistogramPlot],
  templateUrl: './content-heatmaps-statistics-dataset.html',
  styleUrl: './content-heatmaps-statistics-dataset.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
  providers: [Shapes, Scores],
})
export class ContentHeatmapsStatisticsDataset {
  model = input.required<string>();
  dataset = input.required<string>();
  heatmap = input.required<string>();

  private readonly shapesService = inject(Shapes);
  private readonly scoresService = inject(Scores);

  allInputsSelected = computed(() => {
    return this.model() && this.dataset() && this.heatmap();
  });

  averageDropXData = computed(() => {
    const scores = this.scoresService.response();
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
    const scores = this.scoresService.response();
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
    const scores = this.scoresService.response();
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
    const scores = this.scoresService.response();
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
    const scores = this.shapesService.response();
    if (!scores || !scores.areas || scores.areas.length === 0) {
      return [];
    }
    return scores.areas;
  });

  perimetersData = computed(() => {
    const scores = this.shapesService.response();
    if (!scores || !scores.perimeters || scores.perimeters.length === 0) {
      return [];
    }
    return scores.perimeters;
  });

  circularitiesData = computed(() => {
    const scores = this.shapesService.response();
    if (!scores || !scores.circularities || scores.circularities.length === 0) {
      return [];
    }
    return scores.circularities;
  });

  rectangularitiesData = computed(() => {
    const scores = this.shapesService.response();
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

      if (!model || !dataset || !heatmap) {
        return;
      }

      const scoresRequest: HeatmapStatisticsDatasetManyRequest = {
        model: model,
        dataset: dataset,
        heatmap: heatmap,
        percentage: 5,
      };

      this.scoresService.fetchScoresStatisticsDataset(scoresRequest);

      const shapesRequest: ShapeMetricsDatasetRequest = {
        model: model,
        dataset: dataset,
        heatmap: heatmap,
        percentage: 15,
      };

      this.shapesService.fetchShapesStatisticsDataset(shapesRequest);
    });
  }
}
