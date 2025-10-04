import {
  Component,
  input,
  viewChild,
  ElementRef,
  effect,
  ChangeDetectionStrategy,
  OnDestroy,
} from '@angular/core';
import Plotly from 'plotly.js-dist';

@Component({
  selector: 'app-image-statistics-plot',
  imports: [],
  templateUrl: './image-statistics-plot.html',
  styleUrl: './image-statistics-plot.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ImageStatisticsPlot implements OnDestroy {
  xData = input<number[]>([]);
  yData = input<number[]>([]);
  title = input<string>('');
  xAxisLabel = input<string>('');
  yAxisLabel = input<string>('');

  plotContainer = viewChild.required<ElementRef>('plotContainer');
  private plotInitialized = false;

  constructor() {
    effect(() => {
      const xData = this.xData();
      const yData = this.yData();
      this.title();
      this.xAxisLabel();
      this.yAxisLabel();
      const plotElement = this.plotContainer();

      if (plotElement && xData.length > 0 && yData.length > 0) {
        setTimeout(() => this.createPlot(), 0);
      }
    });
  }

  ngOnDestroy(): void {
    const containerElement = this.plotContainer()?.nativeElement;
    if (containerElement && this.plotInitialized) {
      Plotly.purge(containerElement);
    }
  }

  private createPlot(): void {
    const containerElement = this.plotContainer()?.nativeElement;
    if (!containerElement) {
      return;
    }

    const xData = this.xData();
    const yData = this.yData();

    if (xData.length === 0 || yData.length === 0) {
      return;
    }

    const trace: Plotly.Data = {
      x: xData,
      y: yData,
      type: 'scatter',
      mode: 'lines+markers',
      marker: { color: 'blue' },
      line: { color: 'blue' },
    };

    const layout: Partial<Plotly.Layout> = {
      title: { text: this.title() },
      xaxis: {
        title: { text: this.xAxisLabel() },
        autorange: true,
      },
      yaxis: {
        title: { text: this.yAxisLabel() },
        autorange: true,
      },
      autosize: true,
    };

    const config: Partial<Plotly.Config> = {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
    };

    if (this.plotInitialized) {
      Plotly.purge(containerElement);
    }

    Plotly.newPlot(containerElement, [trace], layout, config).then(() => {
      this.plotInitialized = true;
      Plotly.Plots.resize(containerElement);
    });
  }
}
