import { Component } from '@angular/core';
import { ImageComponent } from '../../../components-test/image-component/image-component';

@Component({
  selector: 'app-home-component',
  imports: [ImageComponent],
  templateUrl: './home-component.html',
  styleUrl: './home-component.css'
})
export class HomeComponent {

}
