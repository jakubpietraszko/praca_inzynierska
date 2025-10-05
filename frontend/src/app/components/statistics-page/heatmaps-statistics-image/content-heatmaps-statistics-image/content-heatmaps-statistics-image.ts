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
  GradientStatisticsResponse,
  HeatmapsStatisticsImage,
} from '../heatmaps-statistics-image';
import { Subscription } from 'rxjs';
import { ImageStatisticsPlot } from '../../../shared/image-statistics-plot/image-statistics-plot';
import { DecimalPipe } from '@angular/common';
import { LoadModelsDatasetsImage } from '../../../../directives/load-models-datasets-image';
import { LoadImage } from '../../../../directives/load-image';
import { LoadFourier } from '../../../../directives/load-fourier';

@Component({
  selector: 'app-content-heatmaps-statistics-image',
  imports: [ImageStatisticsPlot, DecimalPipe, LoadModelsDatasetsImage, LoadImage, LoadFourier],
  templateUrl: './content-heatmaps-statistics-image.html',
  styleUrl: './content-heatmaps-statistics-image.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentHeatmapsStatisticsImage implements OnDestroy {
  model = input.required<string>();
  dataset = input.required<string>();
  heatmap = input.required<string>();
  label = input.required<string>();
  image = input.required<string>();

  heatmapsStatisticsService = inject(HeatmapsStatisticsImage);

  insertionScores = signal<GradientStatisticsResponse | null>(null);

  deletionsScores = signal<GradientStatisticsResponse | null>(null);

  insertionSubscription = new Subscription();

  deletionSubscription = new Subscription();

  heatmapGenerated = signal(false);

  allInputsSelected = computed(() => {
    return (
      this.model() &&
      this.dataset() &&
      this.heatmap() &&
      this.label() &&
      this.image()
    );
  });

  insertionXData = computed(() => {
    const scores = this.insertionScores();
    if (
      !scores ||
      !scores.softmax_scores ||
      scores.softmax_scores.length === 0
    ) {
      return [];
    }
    const scoresArray = scores.softmax_scores[0].scores;
    if (scoresArray.length === 0) {
      return [];
    }
    if (scoresArray.length === 1) {
      return [0];
    }
    const step = 100 / (scoresArray.length - 1);
    return Array.from({ length: scoresArray.length }, (_, i) => i * step);
  });

  insertionYData = computed(() => {
    const scores = this.insertionScores();
    if (
      !scores ||
      !scores.softmax_scores ||
      scores.softmax_scores.length === 0
    ) {
      return [];
    }
    return scores.softmax_scores[0].scores;
  });

  deletionXData = computed(() => {
    const scores = this.deletionsScores();
    if (
      !scores ||
      !scores.softmax_scores ||
      scores.softmax_scores.length === 0
    ) {
      return [];
    }
    const scoresArray = scores.softmax_scores[0].scores;
    if (scoresArray.length === 0) {
      return [];
    }
    if (scoresArray.length === 1) {
      return [0];
    }
    const step = 100 / (scoresArray.length - 1);
    return Array.from({ length: scoresArray.length }, (_, i) => i * step);
  });

  deletionYData = computed(() => {
    const scores = this.deletionsScores();
    if (
      !scores ||
      !scores.softmax_scores ||
      scores.softmax_scores.length === 0
    ) {
      return [];
    }
    return scores.softmax_scores[0].scores;
  });

  insertionAUC = computed(() => {
    const xData = this.insertionXData();
    const yData = this.insertionYData();
    if (xData.length === 0 || yData.length === 0) {
      return 0;
    }
    return this.trapezoid(yData, xData);
  });

  deletionAUC = computed(() => {
    const xData = this.deletionXData();
    const yData = this.deletionYData();
    if (xData.length === 0 || yData.length === 0) {
      return 0;
    }
    return this.trapezoid(yData, xData);
  });

  constructor() {
    effect(() => {
      const model = this.model();
      const dataset = this.dataset();
      const heatmap = this.heatmap();
      const label = this.label();
      const image = this.image();

      this.heatmapGenerated.set(false);

      this.insertionSubscription.unsubscribe();
      this.deletionSubscription.unsubscribe();

      this.insertionScores.set(null);
      this.deletionsScores.set(null);

      if (!model || !dataset || !heatmap || !label || !image) {
        return;
      }

      this.insertionSubscription = this.heatmapsStatisticsService
        .getScore({
          model,
          dataset,
          images: [{ label, filename: image, target: null }],
          overwrite: false,
          heatmap,
          threshold: 0,
          threshold_type: 'up',
        })
        .subscribe({
          next: (response) => {
            this.insertionScores.set(response);
            this.heatmapGenerated.set(true);
          },
          error: () => {
            this.insertionScores.set(null);
            this.heatmapGenerated.set(false);
          },
        });

      this.deletionSubscription = this.heatmapsStatisticsService
        .getScore({
          model,
          dataset,
          images: [{ label, filename: image, target: null }],
          overwrite: false,
          heatmap,
          threshold: 0,
          threshold_type: 'down',
        })
        .subscribe({
          next: (response) => {
            this.deletionsScores.set(response);
            this.heatmapGenerated.set(true);
          },
          error: () => {
            this.deletionsScores.set(null);
            this.heatmapGenerated.set(false);
          },
        });
    });
  }

  ngOnDestroy(): void {
    this.insertionSubscription.unsubscribe();
    this.deletionSubscription.unsubscribe();
  }

  linspace(start: number, end: number, num: number): number[] {
    const arr = [];
    const step = (end - start) / (num - 1);
    for (let i = 0; i < num; i++) {
      arr.push(start + step * i);
    }
    return arr;
  }

  trapezoid(y: number[], x: number[]): number {
    let auc = 0;
    for (let i = 1; i < y.length; i++) {
      auc += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
    }
    return auc / 100;
  }
}
