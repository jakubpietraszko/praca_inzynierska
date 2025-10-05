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
  selector: 'app-histogram-plot',
  imports: [],
  templateUrl: './histogram-plot.html',
  styleUrl: './histogram-plot.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class HistogramPlot implements OnDestroy {
  data = input<number[]>([]);
  title = input<string>('');
  xAxisLabel = input<string>('');
  yAxisLabel = input<string>('Frequency');
  nbins = input<number | undefined>(undefined);
  color = input<string>('blue');

  plotContainer = viewChild.required<ElementRef>('plotContainer');
  private plotInitialized = false;

  constructor() {
    effect(() => {
      const data = this.data();
      this.title();
      this.xAxisLabel();
      this.yAxisLabel();
      this.nbins();
      this.color();
      const plotElement = this.plotContainer();

      if (plotElement && data.length > 0) {
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

    const data = this.data();

    if (data.length === 0) {
      return;
    }

    const trace: any = {
      x: data,
      type: 'histogram',
      marker: { 
        color: this.color(),
        line: {
          color: 'white',
          width: 1
        }
      },
      nbinsx: this.nbins(),
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
      bargap: 0.05,
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
