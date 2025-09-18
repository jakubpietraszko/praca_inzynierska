import { Component, inject, model, OnInit, signal } from '@angular/core';
import { HeatmapsService } from '../../../services/heatmaps/heatmaps';
import { AllHeatmapsResponse } from '../../../interfaces/all-heatmaps';
import { FormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { TranslateXaiMethodsPipe } from '../../../pipes/translate-xai-methods-pipe';

@Component({
  selector: 'app-heatmaps-selector',
  imports: [
    FormsModule,
    MatInputModule,
    MatSelectModule,
    MatFormFieldModule,
    TranslateXaiMethodsPipe,
  ],
  templateUrl: './heatmaps-selector.html',
  styleUrl: './heatmaps-selector.css',
})
export class HeatmapsSelector implements OnInit {
  heatmapsService = inject(HeatmapsService);
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
