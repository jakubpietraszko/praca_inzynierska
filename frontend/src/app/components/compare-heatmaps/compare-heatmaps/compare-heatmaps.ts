import { Component, inject, input, OnInit, signal } from '@angular/core';
import { Heatmaps } from '../../../services/heatmaps/heatmaps';
import { AllHeatmapsResponse } from '../../../interfaces/all-heatmaps';
import { ImageTitleTransparent } from '../../image-title-transparent/image-title-transparent';

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

  all_types = signal<string[]>([]);


  heatmapsService = inject(Heatmaps);

  ngOnInit(): void {
    this.heatmapsService
      .allHeatmaps()
      .subscribe((response: AllHeatmapsResponse) => {
        this.all_types.set(response.heatmaps);
      });
  }
}
