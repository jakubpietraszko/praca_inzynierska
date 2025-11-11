import { Component, input, effect, viewChild, ElementRef, ChangeDetectionStrategy, signal, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import * as Plotly from 'plotly.js-dist';
import { environment } from '../../../../environments/environment';
import { FetchImageResponse } from '../../../generated-interfaces/fetch_image';

export interface ScatterPlotPoint {
  x: number;
  y: number;
  color?: string;
  label?: string;
  text?: string;
  hoverText?: string;
  dataset?: string;
  filename?: string;
}

export interface ScatterPlotConfig {
  title?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  showLegend?: boolean;
  width?: number;
  height?: number;
}

@Component({
  selector: 'app-scatter-plot',
  imports: [],
  templateUrl: './scatter-plot.html',
  styleUrl: './scatter-plot.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ScatterPlot {
  data = input.required<ScatterPlotPoint[]>();
  config = input<ScatterPlotConfig>({});
  
  private plotContainer = viewChild.required<ElementRef<HTMLDivElement>>('plotContainer');
  private http = inject(HttpClient);
  
  protected zoomedImageSrc = signal<string | null>(null);
  protected isZoomed = signal<boolean>(false);
  protected imagePosition = signal<{ x: number; y: number }>({ x: 0, y: 0 });
  protected isDragging = signal<boolean>(false);
  private dragStartPos = { x: 0, y: 0 };

  constructor() {
    effect(() => {
      const points = this.data();
      const configuration = this.config();
      const container = this.plotContainer();
      
      if (points && container) {
        this.renderPlot(container.nativeElement, points, configuration);
      }
    });
  }

  private renderPlot(
    container: HTMLDivElement,
    points: ScatterPlotPoint[],
    config: ScatterPlotConfig
  ): void {
    const trace: Partial<Plotly.PlotData> = {
      x: points.map(p => p.x),
      y: points.map(p => p.y),
      mode: 'markers',
      type: 'scatter',
      marker: {
        size: 8,
        color: points.map(p => p.color || '#1f77b4'),
      },
      text: points.map(p => p.text || p.label || ''),
      hovertemplate: '<b>%{text}</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<extra></extra>',
    };

    const layout: Partial<Plotly.Layout> = {
      title: config.title ? { text: config.title } : undefined,
      xaxis: {
        title: config.xAxisLabel ? { text: config.xAxisLabel } : { text: 'X' },
        showgrid: true,
        zeroline: true,
      },
      yaxis: {
        title: config.yAxisLabel ? { text: config.yAxisLabel } : { text: 'Y' },
        showgrid: true,
        zeroline: true,
      },
      showlegend: config.showLegend || false,
      width: config.width,
      height: config.height,
      hovermode: 'closest',
      autosize: true,
      margin: {
        l: 50,
        r: 20,
        t: 40,
        b: 50,
      },
    };

    const plotConfig: Partial<Plotly.Config> = {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    };

    Plotly.newPlot(container, [trace], layout, plotConfig);

    let isMouseDown = false;

    container.addEventListener('mousedown', (event: MouseEvent) => {
      if (event.button === 0) {
        isMouseDown = true;
      }
    });

    container.addEventListener('mouseup', () => {
      isMouseDown = false;
      this.hideZoomedImage();
    });

    (container as any).on('plotly_hover', (data: any) => {
      if (isMouseDown && data.points && data.points.length > 0) {
        const pointIndex = data.points[0].pointIndex;
        const point = points[pointIndex];
        
        if (point.dataset && point.label && point.filename) {
          this.fetchAndShowImage(point.dataset, point.label, point.filename);
        }
      }
    });

    (container as any).on('plotly_click', (data: any) => {
      if (data.points && data.points.length > 0) {
        const pointIndex = data.points[0].pointIndex;
        const point = points[pointIndex];
        
        if (point.dataset && point.label && point.filename) {
          this.fetchAndShowImage(point.dataset, point.label, point.filename);
        }
      }
    });
  }

  private fetchAndShowImage(dataset: string, label: string, filename: string): void {
    this.http.get<FetchImageResponse>(`${environment.apiUrl}/fetch-image`, {
      params: { dataset, label, filename }
    }).subscribe({
      next: (response) => {
        this.zoomedImageSrc.set('data:image/png;base64,' + response.image);
        this.isZoomed.set(true);
      },
      error: (error) => {
        console.error('Error fetching image:', error);
      }
    });
  }

  private hideZoomedImage(): void {
    this.isZoomed.set(false);
    this.zoomedImageSrc.set(null);
    this.imagePosition.set({ x: 0, y: 0 });
    this.isDragging.set(false);
  }

  protected onImageMouseDown(event: MouseEvent): void {
    this.isDragging.set(true);
    const currentPos = this.imagePosition();
    this.dragStartPos = {
      x: event.clientX - currentPos.x,
      y: event.clientY - currentPos.y
    };
    event.preventDefault();
  }

  protected onImageMouseMove(event: MouseEvent): void {
    if (this.isDragging()) {
      this.imagePosition.set({
        x: event.clientX - this.dragStartPos.x,
        y: event.clientY - this.dragStartPos.y
      });
    }
  }

  protected onImageMouseUp(): void {
    this.isDragging.set(false);
  }
}
