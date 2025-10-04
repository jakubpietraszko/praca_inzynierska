import {
  Component,
  inject,
  input,
  OnDestroy,
  OnInit,
  WritableSignal,
  computed,
  signal,
  effect,
} from '@angular/core';
import { HeatmapsWebsocket } from '../../../services/heatmaps-websocket';
import { Heatmaps } from '../../../services/heatmaps';
import { HeatmapsGenerator } from '../../../services/heatmaps-generator';
import { CdkAccordionModule } from '@angular/cdk/accordion';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatButtonModule } from '@angular/material/button';
import { TranslateXaiMethodsPipe } from '../../../pipes/translate-xai-methods-pipe';
import { httpResource } from '@angular/common/http';
import { environment } from '../../../../environments/environment';
import { Subscription } from 'rxjs';

export interface NumeratorDenominatorResponse {
  numerator: Record<string, number>;
  denominator: number;
}

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
})
export class GeneratorList implements OnInit, OnDestroy {
  model = input.required<string>();
  dataset = input.required<string>();
  label = input.required<string>();

  ws = inject(HeatmapsWebsocket);
  heatmapsGenerator = inject(HeatmapsGenerator);
  allHeatmaps = inject(Heatmaps).getAllHeatmaps();
  private processedMessages = new Set<string>();
  private numeratorSignals: Record<string, WritableSignal<number>> = {};
  private wsSubscription?: Subscription;

  numeratorDenominatorResource = httpResource<NumeratorDenominatorResponse>(
    () => ({
      url: `${environment.apiUrl}/numerator-denominator`,
      params: {
        model: this.model(),
        dataset: this.dataset(),
        label: this.label(),
      },
    })
  );

  numeratorDenominatorData = computed<NumeratorDenominatorSignals | null>(
    () => {
      const resourceValue = this.numeratorDenominatorResource.value();
      if (!resourceValue) return null;

      return {
        numerator: this.numeratorSignals,
        denominator: resourceValue.denominator,
      };
    }
  );

  private initializeSignals() {
    const resourceValue = this.numeratorDenominatorResource.value();
    if (resourceValue) {
      for (const [key, value] of Object.entries(resourceValue.numerator)) {
        if (!this.numeratorSignals[key]) {
          this.numeratorSignals[key] = signal(value);
        } else {
          this.numeratorSignals[key].set(value);
        }
      }
    }
  }

  constructor() {
    effect(() => {
      this.initializeSignals();
    });
  }

  ngOnInit() {
    this.ws.connect();

    this.wsSubscription = this.ws.data$.subscribe((wsData) => {
      if (
        wsData &&
        wsData.dataset === this.dataset() &&
        wsData.model === this.model() &&
        wsData.label === this.label() &&
        wsData.heatmap_type &&
        wsData.filename
      ) {
        const messageId = `${wsData.dataset}-${wsData.model}-${wsData.label}-${wsData.heatmap_type}-${wsData.filename}`;

        if (!this.processedMessages.has(messageId)) {
          this.processedMessages.add(messageId);

          if (this.numeratorSignals[wsData.heatmap_type]) {
            this.numeratorSignals[wsData.heatmap_type].update(
              (current) => current + 1
            );
          }
        }
      }
    });
  }

  ngOnDestroy() {
    this.ws.disconnect();
    this.processedMessages.clear();
    this.wsSubscription?.unsubscribe();
  }

  generateHeatmapsByType(heatmapType: string) {
    const messagesToRemove = Array.from(this.processedMessages).filter(
      (messageId) => messageId.includes(`-${heatmapType}-`)
    );
    messagesToRemove.forEach((messageId) =>
      this.processedMessages.delete(messageId)
    );

    this.heatmapsGenerator
      .generateAllHeatmapsByType(
        this.model(),
        this.dataset(),
        this.label(),
        heatmapType
      )
      .subscribe({
        next: () => {},
        error: () => {},
      });
  }

  getProgress(heatmap: string): number {
    const resourceValue = this.numeratorDenominatorData();

    const numerator = this.numeratorSignals[heatmap]?.() || 0;
    const denominator = resourceValue?.denominator || 1;
    return (numerator / denominator) * 100;
  }
}
