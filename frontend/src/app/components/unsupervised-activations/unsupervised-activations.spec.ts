import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UnsupervisedActivations } from './unsupervised-activations';

describe('UnsupervisedActivations', () => {
  let component: UnsupervisedActivations;
  let fixture: ComponentFixture<UnsupervisedActivations>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UnsupervisedActivations]
    })
    .compileComponents();

    fixture = TestBed.createComponent(UnsupervisedActivations);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
