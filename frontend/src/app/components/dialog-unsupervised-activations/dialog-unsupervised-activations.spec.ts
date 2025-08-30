import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DialogUnsupervisedActivations } from './dialog-unsupervised-activations';

describe('DialogUnsupervisedActivations', () => {
  let component: DialogUnsupervisedActivations;
  let fixture: ComponentFixture<DialogUnsupervisedActivations>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DialogUnsupervisedActivations]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DialogUnsupervisedActivations);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
