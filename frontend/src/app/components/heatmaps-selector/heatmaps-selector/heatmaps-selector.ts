import { Component, inject, model, OnInit, signal } from '@angular/core';
import { Heatmaps } from '../../../services/heatmaps/heatmaps';
import { AllHeatmapsResponse } from '../../../interfaces/all-heatmaps';
import { FormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';

@Component({
  selector: 'app-heatmaps-selector',
  imports: [FormsModule, MatInputModule, MatSelectModule, MatFormFieldModule],
  templateUrl: './heatmaps-selector.html',
  styleUrl: './heatmaps-selector.css',
})
export class HeatmapsSelector implements OnInit {
  heatmapsService = inject(Heatmaps);
  heatmaps = signal<string[]>([]);
  selectedHeatmap = model<string | null>(null);

  ngOnInit(): void {
    this.heatmapsService
      .allHeatmaps()
      .subscribe((response: AllHeatmapsResponse) => {
        this.heatmaps.set(response.heatmaps);
      });
  }
}
