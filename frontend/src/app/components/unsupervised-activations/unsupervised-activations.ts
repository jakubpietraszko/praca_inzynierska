import { Component, computed, effect, inject, input, Input, OnDestroy, OnInit } from '@angular/core';
import { LoadActivationImage } from '../../directives/load-activation-image/load-activation-image';
import { DragDropModule } from '@angular/cdk/drag-drop';
import { CommonModule } from '@angular/common';
import { UnsupervisedActivationsService } from '../../services/unsupervised/unsupervised-activations';
import {
  UnsupervisedActivationsRequest,
  UnsupervisedActivationsResponse,
} from '../../interfaces/unsupervised-activations';
import { firstValueFrom, Subscription } from 'rxjs';

interface ImageCoordinate {
  label: string;
  filename: string;
  coordX: number;
  coordY: number;
}

@Component({
  selector: 'app-unsupervised-activations',
  imports: [LoadActivationImage, DragDropModule, CommonModule],
  templateUrl: './unsupervised-activations.html',
  styleUrl: './unsupervised-activations.css',
})
export class UnsupervisedActivations implements OnInit, OnDestroy {
  model = input<string>();
  dataset = input<string>();
  label = input<string>();
  filename = input<string>();
  type = input<string>();

  allInputsSet = computed(
    () => !!this.model() &&
      !!this.dataset() &&
      !!this.label() &&
      !!this.filename() &&
      !!this.type()
  );

  private subscription: Subscription = new Subscription();

  unsupervisedActivationsService = inject(UnsupervisedActivationsService);

  response!: UnsupervisedActivationsResponse;

  constructor() {
    effect( () => {
      if (this.allInputsSet()) {
        this.loadActivations();
      }
    });
  }

    private async loadActivations(): Promise<void> {
    if (
      this.model() &&
      this.dataset() &&
      this.label() &&
      this.filename() &&
      this.type()
    ) {
      const request: UnsupervisedActivationsRequest = {
        model: this.model()!,
        dataset: this.dataset()!,
        label: this.label()!,
        filename: this.filename()!,
        type: this.type()!,
      };

      const result = await firstValueFrom(this.unsupervisedActivationsService.generateActivations(request));
      console.log('Generation result:', result);

      const ref = this.unsupervisedActivationsService
        .getActivations(request)
        .subscribe({
          next: (response: UnsupervisedActivationsResponse) => {
            this.response = response;
            console.log(this.response);
            this.imageCoordinates = this.response.label_filenames_coords;
          },
        });
      this.subscription.add(ref);
    }
  }


  async ngOnInit(): Promise<void> {
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }

  imageCoordinates: ImageCoordinate[] = [];

  onDragEnded(event: any, index: number) {
    const container = event.source.element.nativeElement.parentElement;
    const containerRect = container.getBoundingClientRect();

    const imageElement = event.source.element.nativeElement;
    const imageRect = imageElement.getBoundingClientRect();

    const centerX =
      (imageRect.left + imageRect.width / 2 - containerRect.left) /
      containerRect.width;
    const centerY =
      (imageRect.top + imageRect.height / 2 - containerRect.top) /
      containerRect.height;
  }

  getCurrentCoordinates(): ImageCoordinate[] {
    return this.imageCoordinates;
  }

  resetPositions(): void {
    this.imageCoordinates = [...this.imageCoordinates];
  }
}
