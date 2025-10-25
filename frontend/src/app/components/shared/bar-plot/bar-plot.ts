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
  selector: 'app-bar-plot',
  imports: [],
  templateUrl: './bar-plot.html',
  styleUrl: './bar-plot.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class BarPlot implements OnDestroy {
  labels = input<string[]>([]);
  values = input<number[]>([]);
  title = input<string>('');
  xAxisLabel = input<string>('');
  yAxisLabel = input<string>('Value');
  color = input<string>('blue');

  plotContainer = viewChild.required<ElementRef>('plotContainer');
  private plotInitialized = false;

  constructor() {
    effect(() => {
      const labels = this.labels();
      const values = this.values();
      this.title();
      this.xAxisLabel();
      this.yAxisLabel();
      this.color();
      const plotElement = this.plotContainer();

      if (plotElement && labels.length > 0 && values.length > 0) {
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

    const labels = this.labels();
    const values = this.values();

    if (labels.length === 0 || values.length === 0) {
      return;
    }

    const trace: any = {
      x: labels,
      y: values,
      type: 'bar',
      marker: {
        color: this.color(),
        line: {
          color: 'white',
          width: 1,
        },
      },
    };

    const layout: Partial<Plotly.Layout> = {
      title: { text: this.title() },
      xaxis: {
        title: { text: this.xAxisLabel() },
      },
      yaxis: {
        title: { text: this.yAxisLabel() },
      },
      autosize: true,
      bargap: 0.2,
      margin: {
        b: 100,
      },
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
