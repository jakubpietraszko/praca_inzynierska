import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  input,
  signal,
} from '@angular/core';
import { HeatmapsStatisticsImage } from '../heatmaps-statistics-image';
import { ImageStatisticsPlot } from '../../../shared/image-statistics-plot/image-statistics-plot';
import { DecimalPipe } from '@angular/common';
import { LoadModelsDatasetsImage } from '../../../../directives/load-models-datasets-image';
import { LoadImage } from '../../../../directives/load-image';
import { LoadFourier } from '../../../../directives/load-fourier';
import { LoadFourierDataset } from '../../../../directives/load-fourier-dataset';
import { BarPlot } from '../../../shared/bar-plot/bar-plot';
import { ProbaImage } from '../proba-image';
import { InsertionDeletion } from '../insertion-deletion';

@Component({
  selector: 'app-content-heatmaps-statistics-image',
  imports: [
    ImageStatisticsPlot,
    DecimalPipe,
    LoadModelsDatasetsImage,
    LoadImage,
    LoadFourier,
    LoadFourierDataset,
    BarPlot,
  ],
  providers: [ProbaImage, InsertionDeletion],
  templateUrl: './content-heatmaps-statistics-image.html',
  styleUrl: './content-heatmaps-statistics-image.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentHeatmapsStatisticsImage {
  readonly model = input.required<string>();
  readonly dataset = input.required<string>();
  readonly heatmap = input.required<string>();
  readonly label = input.required<string>();
  readonly image = input.required<string>();

  private readonly heatmapsStatisticsService = inject(HeatmapsStatisticsImage);
  protected readonly probaImageService = inject(ProbaImage);
  protected readonly insertionDeletionService = inject(InsertionDeletion);

  readonly heatmapGenerated = signal(false);

  readonly allInputsSelected = computed(() => {
    return (
      this.model() &&
      this.dataset() &&
      this.heatmap() &&
      this.label() &&
      this.image()
    );
  });

  readonly insertionXData = computed(() => {
    const response = this.insertionDeletionService.response();
    if (
      !response ||
      !response.insertion_scores ||
      response.insertion_scores.length === 0
    ) {
      return [];
    }
    const scoresArray = response.insertion_scores[0].scores;
    if (scoresArray.length === 0) {
      return [];
    }
    if (scoresArray.length === 1) {
      return [0];
    }
    const step = 100 / (scoresArray.length - 1);
    return Array.from({ length: scoresArray.length }, (_, i) => i * step);
  });

  readonly insertionYData = computed(() => {
    const response = this.insertionDeletionService.response();
    if (
      !response ||
      !response.insertion_scores ||
      response.insertion_scores.length === 0
    ) {
      return [];
    }
    return response.insertion_scores[0].scores;
  });

  readonly deletionXData = computed(() => {
    const response = this.insertionDeletionService.response();
    if (
      !response ||
      !response.deletion_scores ||
      response.deletion_scores.length === 0
    ) {
      return [];
    }
    const scoresArray = response.deletion_scores[0].scores;
    if (scoresArray.length === 0) {
      return [];
    }
    if (scoresArray.length === 1) {
      return [0];
    }
    const step = 100 / (scoresArray.length - 1);
    return Array.from({ length: scoresArray.length }, (_, i) => i * step);
  });

  readonly deletionYData = computed(() => {
    const response = this.insertionDeletionService.response();
    if (
      !response ||
      !response.deletion_scores ||
      response.deletion_scores.length === 0
    ) {
      return [];
    }
    return response.deletion_scores[0].scores;
  });

  readonly insertionAUC = computed(() => {
    const response = this.insertionDeletionService.response();
    return response?.insertion_auc ?? 0;
  });

  readonly deletionAUC = computed(() => {
    const response = this.insertionDeletionService.response();
    return response?.deletion_auc ?? 0;
  });

  readonly probaClasses = computed(() => {
    const proba = this.probaImageService.response();
    return proba?.classes ?? [];
  });

  readonly probaProbabilities = computed(() => {
    const proba = this.probaImageService.response();
    return proba?.probabilities ?? [];
  });

  constructor() {
    effect(() => {
      const model = this.model();
      const dataset = this.dataset();
      const heatmap = this.heatmap();
      const label = this.label();
      const image = this.image();

      this.heatmapGenerated.set(false);

      if (!model || !dataset || !heatmap || !label || !image) {
        return;
      }

      this.probaImageService.fetchProbaForImage({
        model,
        dataset,
        label,
        filename: image,
      });

      this.insertionDeletionService.fetchInsertionDeletion({
        model,
        dataset,
        images: [{ label, filename: image, target: null }],
        heatmap,
        threshold: 0,
      });

      this.heatmapGenerated.set(true);
    });
  }
}
