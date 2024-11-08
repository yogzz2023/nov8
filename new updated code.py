Replace your current process_udp_data and handle_single_measurement methods in the KalmanFilterGUI class with:

def process_udp_data(self, measurement):
    try:
        # Convert string measurements to float
        measurement = list(map(float, measurement))
        
        # Extract coordinates and time
        mr = measurement[10]  # Range
        ma = measurement[11]  # Azimuth
        me = measurement[12]  # Elevation
        mt = measurement[13]  # Time
        md = measurement[14]  # Doppler
        
        # Convert to Cartesian coordinates
        x, y, z = sph2cart(ma, me, mr)
        
        # Create measurement tuple in the expected format
        formatted_measurement = (mr, ma, me, mt, md, x, y, z)
        
        # Process the measurement
        self.handle_real_time_measurement(formatted_measurement)
        
    except Exception as e:
        print(f"Error processing UDP")

def handle_real_time_measurement(self, measurement):
    if not hasattr(self, 'real_time_tracks'):
        self.real_time_tracks = []
        
    track_mode = self.track_mode_combo.currentText()
    association_type = "JPDA" if self.jpda_radio.isChecked() else "Munkres"
    filter_option = self.filter_mode

    # Group the single measurement
    measurement_group = [measurement]
    
    # If this is the first measurement, initialize tracks
    if not self.real_time_tracks:
        new_track = {
            'track_id': 0,
            'measurements': [(measurement, 'Poss1')],
            'current_state': 'Poss1',
            'Sf': [],
            'Sp': [],
            'Pp': [],
            'Pf': []
        }
        self.real_time_tracks.append(new_track)
    else:
        # Perform track association and update
        if association_type == "JPDA":
            clusters, best_reports, _, _ = perform_jpda(self.real_time_tracks, measurement_group, CVFilter())
        else:
            best_reports = perform_munkres(self.real_time_tracks, measurement_group, CVFilter())
            
        # Update tracks with new measurement
        assigned = False
        for track_id, report in best_reports:
            track = self.real_time_tracks[track_id]
            track['measurements'].append((measurement, track['current_state']))
            assigned = True
            
        if not assigned:
            # Create new track
            new_track = {
                'track_id': len(self.real_time_tracks),
                'measurements': [(measurement, 'Poss1')],
                'current_state': 'Poss1',
                'Sf': [],
                'Sp': [],
                'Pp': [],
                'Pf': []
            }
            self.real_time_tracks.append(new_track)
    
    # Update the plot with real-time data
    self.tracks = self.real_time_tracks
    self.update_plot()
    self.update_track_selection()


Replace your current start_udp_receiver method with:
    
def start_udp_receiver(self):
    try:
        if self.udp_thread is None or not self.udp_thread.is_alive():
            self.udp_thread = threading.Thread(
                target=udp_receiver, 
                args=(5005, self.data_signal), 
                daemon=True
            )
            self.udp_thread.start()
            print("UDP receiver started successfully")
            
            # Enable real-time plotting
            self.plot_type_combo.setCurrentText("All Modes")  # Set default view
            self.receive_udp_button.setText("Stop UDP")
            self.receive_udp_button.setStyleSheet("background-color: #FF4444;")
        else:
            print("UDP receiver is already running")
    except Exception as e:
        print(f"Error starting UDP receiver: {e}")


Add this new method to your KalmanFilterGUI class:
    
    
def update_real_time_plot(self):
    if hasattr(self, 'real_time_tracks') and self.real_time_tracks:
        plot_type = self.plot_type_combo.currentText()
        
        # Clear the current plot
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        
        # Plot the real-time tracks
        if plot_type == "All Modes":
            self.plot_all_modes(self.real_time_tracks, ax)
        elif plot_type == "PPI":
            self.plot_ppi(self.real_time_tracks, ax)
        elif plot_type == "RHI":
            self.plot_rhi(self.real_time_tracks, ax)
        else:
            plot_measurements(self.real_time_tracks, ax, plot_type, self.selected_track_ids)
            
        self.canvas.draw()


Replace your current update_plot method with:
    
def update_plot(self):
    tracks_to_plot = self.real_time_tracks if hasattr(self, 'real_time_tracks') else self.tracks
    
    if not tracks_to_plot:
        print("No tracks to plot.")
        return

    plot_type = self.plot_type_combo.currentText()
    
    self.canvas.figure.clear()
    ax = self.canvas.figure.add_subplot(111)
    
    if plot_type == "All Modes":
        self.plot_all_modes(tracks_to_plot, ax)
    elif plot_type == "PPI":
        self.plot_ppi(tracks_to_plot, ax)
    elif plot_type == "RHI":
        self.plot_rhi(tracks_to_plot, ax)
    else:
        plot_measurements(tracks_to_plot, ax, plot_type, self.selected_track_ids)
        
    self.canvas.draw()
