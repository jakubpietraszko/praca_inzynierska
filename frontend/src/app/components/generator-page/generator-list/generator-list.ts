import {
  Component,
  inject,
  input,
  WritableSignal,
  computed,
  signal,
  effect,
  ChangeDetectionStrategy,
  DestroyRef,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { HeatmapsWebsocket } from '../heatmaps-websocket';
import { HeatmapsGenerator } from '../heatmaps-generator';
import { CdkAccordionModule } from '@angular/cdk/accordion';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatButtonModule } from '@angular/material/button';
import { TranslateXaiMethodsPipe } from '../../../pipes/translate-xai-methods-pipe';
import { httpResource } from '@angular/common/http';
import { environment } from '../../../../environments/environment';
import { AllHeatmaps } from '../../../services/all-heatmaps';
import { HeatmapNotification } from '../../../generated-interfaces/notification';
import { NumeratorDenominator } from '../../../generated-interfaces/numerator_denominator';

export interface NumeratorDenominatorSignals {
  numerator: Record<string, WritableSignal<number>>;
  denominator: number;
}

@Component({
  selector: 'app-generator-list',
  imports: [
    CdkAccordionModule,
    MatProgressBarModule,
    MatButtonModule,
    TranslateXaiMethodsPipe,
  ],
  templateUrl: './generator-list.html',
  styleUrl: './generator-list.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: true,
})
export class GeneratorList {
  private readonly ws = inject(HeatmapsWebsocket);
  private readonly heatmapsGenerator = inject(HeatmapsGenerator);
  private readonly allHeatmaps = inject(AllHeatmaps);

  private readonly destroyRef = inject(DestroyRef);

  readonly model = input.required<string>();
  readonly dataset = input.required<string>();
  readonly label = input.required<string>();

  readonly heatmaps = this.allHeatmaps.heatmaps;

  private readonly numeratorSignals = signal<Record<string, WritableSignal<number>>>({});

  readonly numeratorDenominatorResource = httpResource<NumeratorDenominator>(() => ({
    url: `${environment.apiUrl}/numerator-denominator`,
    params: {
      model: this.model(),
      dataset: this.dataset(),
      label: this.label(),
    },
  }));

  readonly numeratorDenominatorData = computed<NumeratorDenominatorSignals | null>(
    () => {
      const resourceValue = this.numeratorDenominatorResource.value();
      if (!resourceValue) return null;

      return {
        numerator: this.numeratorSignals(),
        denominator: resourceValue.denominator,
      };
    },
  );

  constructor() {
    effect(() => {
      this.numeratorDenominatorResource.value();
      this.initializeSignals();
    });

    effect(() => {
      this.ws.connect();
      this.setupWebSocketListener();

      return () => {
        this.ws.disconnect();
      };
    });
  }

  private initializeSignals(): void {
    const resourceValue = this.numeratorDenominatorResource.value();
    if (!resourceValue) return;

    const signals = this.numeratorSignals();
    for (const [key, value] of Object.entries(resourceValue.numerator)) {
      if (!signals[key]) {
        signals[key] = signal(value);
      } else {
        signals[key].set(value);
      }
    }
  }

  private setupWebSocketListener(): void {
    this.ws.data$
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((wsData: HeatmapNotification) => {
        if (this.isRelevantMessage(wsData)) {
          const signals = this.numeratorSignals();
          if (signals[wsData.heatmap_type]) {
            signals[wsData.heatmap_type].update((currentValue) => currentValue + 1);
          }
        }
      });
  }

  private isRelevantMessage(wsData: HeatmapNotification): boolean {
    return (
      wsData?.dataset === this.dataset() &&
      wsData?.model === this.model() &&
      wsData?.label === this.label() &&
      !!wsData?.heatmap_type &&
      !!wsData?.filename
    );
  }

  generateHeatmapsByType(heatmapType: string): void {
    this.heatmapsGenerator
      .generateAllHeatmapsByType(
        this.model(),
        this.dataset(),
        this.label(),
        heatmapType,
      )
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe();
  }

  getProgress(heatmap: string): number {
    const data = this.numeratorDenominatorData();
    const signals = this.numeratorSignals();
    const numerator = signals[heatmap]?.() || 0;
    const denominator = data?.denominator || 1;
    return (numerator / denominator) * 100;
  }
}
