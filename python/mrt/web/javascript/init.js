import { roomName, update_yaml_configurations } from './utils.js';

const yamlInitSocket = new WebSocket(
  'ws://'
  + window.location.host
  + '/ws/web/yaml/init/'
  + roomName
  + '/'
);

yamlInitSocket.onopen = function(e) {
  yamlInitSocket.send(null);
};

yamlInitSocket.addEventListener(
  'message', update_yaml_configurations);

const yamlResetter = document.querySelector('#yaml-resetter');

yamlResetter.onclick = function(e) {
  yamlInitSocket.send(null);
};
