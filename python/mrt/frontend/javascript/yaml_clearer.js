import { roomName, create_socket, update_console } from './utils.js';

const yamlClearSocket = create_socket("yaml/clear/")

yamlClearSocket.onmessage = function(e) {
  const data = JSON.parse(e.data);
  for (const [stage, stage_data] of Object.entries(data)) {
    for (const [attr, value] of Object.entries(stage_data)) {
      const id = '#' + stage + '_' + attr;
      document.querySelector(id).value = '';
    }
  }
  update_console("yaml parameters cleared.");
};

const yamlClearer = document.querySelector('#yaml-clearer');
yamlClearer.onclick = function(e) {
  yamlClearSocket.send(null)
};
