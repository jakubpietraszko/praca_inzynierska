import {
  ChangeDetectionStrategy,
  Component,
  inject,
  input,
  OnDestroy,
  OnInit,
  signal,
} from '@angular/core';
import {
  CompareHeatmapsRequest,
  CompareHeatmapsResponse,
  GeneratorCompareHeatmaps,
} from '../generator-compare-heatmaps';
import { Subscription } from 'rxjs';
import { httpResource } from '@angular/common/http';
import { environment } from '../../../../../environments/environment';
import { ImageTitleTransparent } from '../../../shared/image-title-transparent/image-title-transparent';

@Component({
  selector: 'app-content-compare-heatmaps',
  imports: [ImageTitleTransparent],
  templateUrl: './content-compare-heatmaps.html',
  styleUrl: './content-compare-heatmaps.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentCompareHeatmaps implements OnInit, OnDestroy {
  model = input.required<string>();
  dataset = input.required<string>();
  label = input.required<string>();
  filename = input.required<string>();

  response = signal<CompareHeatmapsResponse | null>(null);

  private subscription = new Subscription();

  compareHeatmapsService = inject(GeneratorCompareHeatmaps);

  allHeatmaps = httpResource<{ heatmaps: string[] }>(
    () => `${environment.apiUrl}/heatmap/all-heatmaps`,
  );

  ngOnInit(): void {
    const request: CompareHeatmapsRequest = {
      model: this.model(),
      dataset: this.dataset(),
      images: [
        {
          label: this.label(),
          filename: this.filename(),
          target: null,
        },
      ],
      overwrite: false,
    };

    this.subscription.add(
      this.compareHeatmapsService
        .compareHeatmaps(request)
        .subscribe((response: CompareHeatmapsResponse) => {
          this.response.set(response);
        }),
    );
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }
}
