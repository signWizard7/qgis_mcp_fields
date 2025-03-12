#!/usr/bin/env python3
"""
Ejemplo de uso del cliente QGIS MCP en un script controlado por un LLM
"""

from qgis_mcp_client import QgisMCPClient
import json
import time

def print_json(data):
    """Imprime datos JSON formateados"""
    print(json.dumps(data, indent=2))

def main():
    # Conectar al servidor QGIS MCP
    client = QgisMCPClient(host='localhost', port=9876)
    if not client.connect():
        print("No se pudo conectar al servidor QGIS MCP")
        return
    
    try:
        # Verificar conexión con ping
        print("Verificando conexión...")
        response = client.ping()
        if response and response.get("status") == "success":
            print("Conexión exitosa")
        else:
            print("Error de conexión")
            return
        
        # Obtener información de QGIS
        print("\nInformación de QGIS:")
        qgis_info = client.get_qgis_info()
        print_json(qgis_info)
        
        # Load project
        print("\nLoad project")
        load_project = client.load_project("C:/Users/jjsan/OneDrive/Consultoria/Finalizados/electoral_maps/thailand_2007/thailand_2007.qgz")
        print_json(load_project)

        # Obtener información del proyecto actual
        print("\nInformación del proyecto:")
        project_info = client.get_project_info()
        print_json(project_info)

        # Zoom to layer
        print("\nZoom to first layer")
        first_layer = project_info["result"]["layers"][0]["id"]
        zoom_layer = client.zoom_to_layer(first_layer)
        print_json(zoom_layer)

        # Render Map to file
        print("\nRendering image")
        render_map = client.render_map("C:/Users/jjsan/OneDrive/Consultoria/Finalizados/electoral_maps/thailand_2007/map.png")
        print_json(render_map)
        
    except Exception:
        print("Error ejecutando comandos")

if __name__ == "__main__":
    main()