import { Component } from '@angular/core';
import { LoadImageDirective } from '../../../directives/load-image/load-image-directive';
import { UnsupervisedActivations } from '../../unsupervised-activations/unsupervised-activations';
import { AllAvailableMethods } from '../../all-available-methods/all-available-methods';
import { ImageTitleTransparent } from '../../image-title-transparent/image-title-transparent';
import { CdkDrag } from '@angular/cdk/drag-drop';
import { CompareHeatmaps } from '../../compare-heatmaps/compare-heatmaps/compare-heatmaps';

@Component({
  selector: 'app-home-component',
  imports: [
    UnsupervisedActivations,
    AllAvailableMethods,
    ImageTitleTransparent,
    CompareHeatmaps
  ],
  templateUrl: './home-component.html',
  styleUrl: './home-component.css',
})
export class HomeComponent {}
