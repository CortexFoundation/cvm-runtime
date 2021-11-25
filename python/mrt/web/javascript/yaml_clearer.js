import { roomName } from './utils.js';

const yamlClearSocket = new WebSocket(
  'ws://'
  + window.location.host
  + '/ws/web/yaml/clear/'
  + roomName
  + '/'
);

yamlClearSocket.onmessage = function(e) {
  const data = JSON.parse(e.data);
  for (const [stage, stage_data] of Object.entries(data)) {
    for (const [attr, value] of Object.entries(stage_data)) {
      const id = '#' + stage + '_' + attr;
      document.querySelector(id).value = '';
    }
  }
};

const yamlClearer = document.querySelector('#yaml-clearer');
yamlClearer.onclick = function(e) {
  yamlClearSocket.send(null)
};
