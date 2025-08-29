import { Component } from '@angular/core';
import { LoadImageDirective } from '../../../directives/load-image/load-image-directive';
import { UnsupervisedActivations } from '../../unsupervised-activations/unsupervised-activations';

@Component({
  selector: 'app-home-component',
  imports: [UnsupervisedActivations],
  templateUrl: './home-component.html',
  styleUrl: './home-component.css'
})
export class HomeComponent {

}
