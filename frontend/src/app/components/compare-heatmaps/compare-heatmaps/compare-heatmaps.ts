import {
  Component,
  computed,
  effect,
  inject,
  input,
  OnInit,
  signal,
} from '@angular/core';
import { HeatmapsService } from '../../../services/heatmaps/heatmaps';
import {
  AllHeatmapsResponse,
  HeatmapRequest,
} from '../../../interfaces/all-heatmaps';
import { ImageTitleTransparent } from '../../image-title-transparent/image-title-transparent';
import { firstValueFrom } from 'rxjs';

@Component({
  selector: 'app-compare-heatmaps',
  imports: [ImageTitleTransparent],
  templateUrl: './compare-heatmaps.html',
  styleUrl: './compare-heatmaps.css',
})
export class CompareHeatmaps implements OnInit {
  dataset = input<string>();
  label = input<string>();
  filename = input<string>();
  model = input<string>();

  allInputsSet = computed(
    () =>
      !!this.dataset() && !!this.label() && !!this.filename() && !!this.model()
  );

  all_types = signal<string[]>([]);
  imagesLoaded = signal<boolean>(false);

  constructor() {
    effect(() => {
      if (this.allInputsSet()) {
        this.loadHeatmaps();
      }
    });
  }

  private async loadHeatmaps(): Promise<void> {
    if (this.dataset() && this.label() && this.filename() && this.model()) {
      const request: HeatmapRequest = {
        model: this.model()!,
        dataset: this.dataset()!,
        images: [
          {
            label: this.label()!,
            filename: this.filename()!,
            target: null,
          },
        ],
        overwrite: false,
      };

      const result = await firstValueFrom(
        this.heatmapsService.generateHeatmap(request)
      );

      this.imagesLoaded.set(true);
    }
  }

  heatmapsService = inject(HeatmapsService);

  ngOnInit(): void {
    this.heatmapsService
      .allHeatmaps()
      .subscribe((response: AllHeatmapsResponse) => {
        this.all_types.set(response.heatmaps);
      });
  }
}
