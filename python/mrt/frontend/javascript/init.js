import { 
  roomName, update_yaml_configurations, update_console,
    create_socket } from './utils.js';

const yamlInitSocket = create_socket("yaml/init/");

yamlInitSocket.onopen = function(e) {
  yamlInitSocket.send(null);
};

yamlInitSocket.onmessage = function(e) {
  update_yaml_configurations(e);
  update_console("yaml parameters initialized.");
}

const yamlResetter = document.querySelector('#yaml-resetter');

yamlResetter.onclick = function(e) {
  yamlInitSocket.send(null);
};
