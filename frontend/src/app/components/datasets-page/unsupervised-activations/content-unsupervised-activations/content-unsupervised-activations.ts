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
  GeneratorUnsupervisedActivations,
  UnsupervisedActivationsRequest,
  UnsupervisedActivationsResponse,
} from '../generator-unsupervised-activations';
import { Subscription } from 'rxjs';
import { DragDropModule } from '@angular/cdk/drag-drop';
import { LoadActivationImage } from '../../../../directives/load-activation-image';

@Component({
  selector: 'app-content-unsupervised-activations',
  imports: [LoadActivationImage, DragDropModule],
  templateUrl: './content-unsupervised-activations.html',
  styleUrl: './content-unsupervised-activations.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentUnsupervisedActivations implements OnInit, OnDestroy {
  model = input.required<string>();
  dataset = input.required<string>();
  label = input.required<string>();
  filename = input.required<string>();
  type = signal<string>('tsne');

  response = signal<UnsupervisedActivationsResponse | null>(null);

  private subscription = new Subscription();

  unsupervisedActivationsService = inject(GeneratorUnsupervisedActivations);

  ngOnInit(): void {
    const request: UnsupervisedActivationsRequest = {
      model: this.model(),
      dataset: this.dataset(),
      label: this.label(),
      filename: this.filename(),
      type: this.type(),
    };

    this.subscription.add(
      this.unsupervisedActivationsService
        .unsupervisedActivations(request)
        .subscribe((response: UnsupervisedActivationsResponse) => {
          this.response.set(response);
        }),
    );
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }
}
