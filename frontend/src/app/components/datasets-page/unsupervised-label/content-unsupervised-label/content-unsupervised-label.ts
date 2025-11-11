import { Component, input, computed, ChangeDetectionStrategy } from '@angular/core';
import { UnsupervisedLabelResponse } from '../../../../generated-interfaces/unsupervised_label';
import { ScatterPlot, ScatterPlotPoint, ScatterPlotConfig } from '../../../shared/scatter-plot/scatter-plot';

@Component({
  selector: 'app-content-unsupervised-label',
  imports: [ScatterPlot],
  templateUrl: './content-unsupervised-label.html',
  styleUrl: './content-unsupervised-label.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentUnsupervisedLabel {
  response = input<UnsupervisedLabelResponse | null>(null);
  dataset = input.required<string>();
  label = input.required<string>();

  protected plotData = computed<ScatterPlotPoint[]>(() => {
    const data = this.response();
    const datasetName = this.dataset();
    const labelName = this.label();
    if (!data) return [];

    return data.items.map((item) => ({
      x: item.coords.x,
      y: item.coords.y,
      color: `rgb(${item.color.r}, ${item.color.g}, ${item.color.b})`,
      label: labelName,
      text: item.filename,
      hoverText: item.filename,
      dataset: datasetName,
      filename: item.filename,
    }));
  });

  protected plotConfig: ScatterPlotConfig = {
    title: 'Redukcja wymiarowości dla etykiety',
    xAxisLabel: 'Zmienna 1',
    yAxisLabel: 'Zmienna 2',
    showLegend: false,
  };
}
