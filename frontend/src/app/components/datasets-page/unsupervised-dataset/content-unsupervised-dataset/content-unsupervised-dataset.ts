import { Component, input, computed, ChangeDetectionStrategy } from '@angular/core';
import { UnsupervisedDatasetResponse } from '../../../../generated-interfaces/unsupervised_dataset';
import { ScatterPlot, ScatterPlotPoint, ScatterPlotConfig } from '../../../shared/scatter-plot/scatter-plot';

@Component({
  selector: 'app-content-unsupervised-dataset',
  imports: [ScatterPlot],
  templateUrl: './content-unsupervised-dataset.html',
  styleUrl: './content-unsupervised-dataset.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentUnsupervisedDataset {
  response = input<UnsupervisedDatasetResponse | null>(null);
  dataset = input.required<string>();

  protected plotData = computed<ScatterPlotPoint[]>(() => {
    const data = this.response();
    const datasetName = this.dataset();
    if (!data) return [];

    return data.items.map((item) => ({
      x: item.coords.x,
      y: item.coords.y,
      color: `rgb(${item.color.r}, ${item.color.g}, ${item.color.b})`,
      label: item.label,
      text: `${item.label} - ${item.filename}`,
      hoverText: `${item.label} - ${item.filename}`,
      dataset: datasetName,
      filename: item.filename,
    }));
  });

  protected plotConfig: ScatterPlotConfig = {
    title: 'Redukcja wymiarowości zbioru danych',
    xAxisLabel: 'Zmienna 1',
    yAxisLabel: 'Zmienna 2',
    showLegend: false,
  };
}
