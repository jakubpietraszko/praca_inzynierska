import { Component } from '@angular/core';
import { LoadImageDirective } from '../../../directives/load-image/load-image-directive';

@Component({
  selector: 'app-home-component',
  imports: [LoadImageDirective],
  templateUrl: './home-component.html',
  styleUrl: './home-component.css'
})
export class HomeComponent {

}
